//! Windows worker isolation primitives.

use std::io;
use std::process::Child;

/// Process-tree and outbound-network isolation for one analysis worker.
pub struct Sandbox {
    #[cfg(windows)]
    job: windows_impl::Job,
    #[cfg(windows)]
    _firewall: windows_impl::Firewall,
}

impl Sandbox {
    /// Attach isolation to a spawned worker.
    ///
    /// Windows Job Objects enforce process-tree lifetime and the aggregate
    /// memory cap. Windows Firewall rules deny outbound traffic for the
    /// worker executable and radare2 binaries resolved from `PATH`.
    pub fn attach(child: &Child, memory_mib: u64) -> io::Result<Self> {
        #[cfg(windows)]
        {
            let job = windows_impl::Job::attach(child, memory_mib)?;
            let firewall = match windows_impl::Firewall::install() {
                Ok(firewall) => firewall,
                Err(error) => {
                    job.terminate();
                    return Err(error);
                }
            };
            return Ok(Self {
                job,
                _firewall: firewall,
            });
        }
        #[cfg(not(windows))]
        {
            let _ = (child, memory_mib);
            Ok(Self {})
        }
    }

    /// Terminate every process assigned to this worker sandbox.
    pub fn terminate(&self) {
        #[cfg(windows)]
        self.job.terminate();
    }
}

#[cfg(windows)]
#[allow(unsafe_code)]
mod windows_impl {
    use std::io;
    use std::mem::size_of;
    use std::os::windows::io::AsRawHandle;
    use std::path::PathBuf;
    use std::process::{Child, Command};

    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_JOB_MEMORY,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JobObjectExtendedLimitInformation, SetInformationJobObject, TerminateJobObject,
    };

    pub struct Job {
        handle: HANDLE,
    }

    impl Job {
        pub fn attach(child: &Child, memory_mib: u64) -> io::Result<Self> {
            let handle = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
            if handle.is_null() {
                return Err(io::Error::last_os_error());
            }

            let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            limits.BasicLimitInformation.LimitFlags =
                JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_JOB_MEMORY;
            limits.JobMemoryLimit = usize::try_from(memory_mib)
                .ok()
                .and_then(|mib| mib.checked_mul(1024 * 1024))
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "memory limit overflow")
                })?;
            let configured = unsafe {
                SetInformationJobObject(
                    handle,
                    JobObjectExtendedLimitInformation,
                    (&limits as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
                    u32::try_from(size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>())
                        .unwrap_or(u32::MAX),
                )
            };
            if configured == 0 {
                let error = io::Error::last_os_error();
                unsafe { CloseHandle(handle) };
                return Err(error);
            }

            let assigned =
                unsafe { AssignProcessToJobObject(handle, child.as_raw_handle() as HANDLE) };
            if assigned == 0 {
                let error = io::Error::last_os_error();
                unsafe { CloseHandle(handle) };
                return Err(error);
            }
            Ok(Self { handle })
        }

        pub fn terminate(&self) {
            unsafe { TerminateJobObject(self.handle, 1) };
        }
    }

    impl Drop for Job {
        fn drop(&mut self) {
            unsafe { CloseHandle(self.handle) };
        }
    }

    pub struct Firewall {
        rules: Vec<String>,
    }

    impl Firewall {
        pub fn install() -> io::Result<Self> {
            let mut programs = vec![std::env::current_exe()?];
            programs.extend(resolve_radare2());
            programs.sort();
            programs.dedup();

            let prefix = format!("r2smt-analysis-{}", std::process::id());
            let mut rules = Vec::with_capacity(programs.len());
            for (index, program) in programs.iter().enumerate() {
                let name = format!("{prefix}-{index}");
                let status = Command::new("netsh")
                    .args([
                        "advfirewall",
                        "firewall",
                        "add",
                        "rule",
                        &format!("name={name}"),
                        "dir=out",
                        "action=block",
                        &format!("program={}", program.display()),
                        "enable=yes",
                        "profile=any",
                    ])
                    .status()?;
                if !status.success() {
                    let firewall = Self { rules };
                    drop(firewall);
                    return Err(io::Error::other(format!(
                        "netsh failed to block {}",
                        program.display()
                    )));
                }
                rules.push(name);
            }
            Ok(Self { rules })
        }
    }

    impl Drop for Firewall {
        fn drop(&mut self) {
            for name in &self.rules {
                let _ = Command::new("netsh")
                    .args([
                        "advfirewall",
                        "firewall",
                        "delete",
                        "rule",
                        &format!("name={name}"),
                    ])
                    .status();
            }
        }
    }

    fn resolve_radare2() -> Vec<PathBuf> {
        ["radare2.exe", "r2.exe"]
            .into_iter()
            .flat_map(|name| {
                let output = Command::new("where.exe").arg(name).output().ok()?;
                output.status.success().then_some(output.stdout)
            })
            .flat_map(|bytes| {
                String::from_utf8_lossy(&bytes)
                    .lines()
                    .map(PathBuf::from)
                    .collect::<Vec<_>>()
            })
            .filter(|path| path.is_file())
            .collect()
    }
}
