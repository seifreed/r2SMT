//! Smoke test for Windows Job Object and firewall isolation.

#[cfg(windows)]
use std::process::Command;
#[cfg(windows)]
use std::thread;
#[cfg(windows)]
use std::time::Duration;

#[cfg(windows)]
use r2smt_windows_sandbox::Sandbox;

#[cfg(windows)]
fn main() {
    let child = Command::new("cmd.exe")
        .args(["/c", "ping.exe -n 30 127.0.0.1 > NUL"])
        .spawn()
        .expect("spawn sandbox smoke child");
    let sandbox = Sandbox::attach(&child, 128).expect("attach Windows sandbox");
    thread::sleep(Duration::from_millis(100));
    sandbox.terminate();
    println!("windows sandbox smoke ok");
}

#[cfg(not(windows))]
fn main() {
    println!("windows sandbox smoke skipped on non-Windows");
}
