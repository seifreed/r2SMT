# r2pm package manifest for r2SMT.
#
# Install with:  r2pm -ci r2smt
# Remove with:   r2pm -u r2smt
#
# After install, launch r2 with:
#   r2 -i $(r2pm -H R2PM_PLUGDIR)/r2smt.r2 <binary>
# and use the macros `$r2smt-solve`, `$r2smt-annotate`, `$r2smt-patch`
# from the r2 prompt.
#
# Install strategy:
#
# - `USE_PREBUILT=1` (default): pull the published release tarball for
#   the host triple from GitHub and unpack the prebuilt `r2smt` into
#   `R2PM_BINDIR`. Falls through to `install-from-source` on any
#   download failure (HTTP 404, host offline, missing triple).
#
# - `USE_PREBUILT=0`: always build from source via `cargo build
#   --release`. Required for developer setups and unsupported triples.

R2PM_BEGIN
R2PM_GIT https://github.com/seifreed/r2SMT
R2PM_DESC "SMT-assisted opaque-predicate deobfuscator for radare2"
R2PM_LICENSE "MIT OR Apache-2.0"
R2PM_TAGS "deobfuscation smt z3 opaque-predicate"
R2PM_NEEDS "rust cargo z3"
R2PM_END

R2PM_BINDIR := $(shell r2pm -H R2PM_BINDIR 2>/dev/null)
R2PM_PLUGDIR := $(shell r2pm -H R2PM_PLUGDIR 2>/dev/null)

ifeq ($(R2PM_BINDIR),)
R2PM_BINDIR := $(HOME)/.local/share/radare2/prefix/bin
endif
ifeq ($(R2PM_PLUGDIR),)
R2PM_PLUGDIR := $(HOME)/.local/share/radare2/plugins
endif

CARGO        ?= cargo
USE_PREBUILT ?= 1
VERSION      ?= 0.1.0
RELEASE_BASE := https://github.com/seifreed/r2SMT/releases/download/v$(VERSION)

# Derive the GitHub release triple from `uname`. Maps `Darwin arm64` →
# `aarch64-apple-darwin`, `Linux x86_64` → `x86_64-unknown-linux-gnu`,
# `Linux aarch64` → `aarch64-unknown-linux-gnu`. Any other shape falls
# through to source build because no release tarball exists for it.
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
ARCH    := $(if $(filter arm64,$(UNAME_M)),aarch64,$(UNAME_M))

ifeq ($(UNAME_S),Darwin)
TRIPLE := $(ARCH)-apple-darwin
else ifeq ($(UNAME_S),Linux)
TRIPLE := $(ARCH)-unknown-linux-gnu
else
TRIPLE :=
endif

TARBALL := r2smt-v$(VERSION)-$(TRIPLE).tar.gz

.PHONY: install install-from-source install-prebuilt install-macros uninstall test clean

install: install-macros
ifeq ($(USE_PREBUILT),1)
	@$(MAKE) -f $(firstword $(MAKEFILE_LIST)) install-prebuilt \
		|| $(MAKE) -f $(firstword $(MAKEFILE_LIST)) install-from-source
else
	@$(MAKE) -f $(firstword $(MAKEFILE_LIST)) install-from-source
endif
	@echo "[r2smt] installed binary: $(R2PM_BINDIR)/r2smt"
	@echo "[r2smt] installed macros: $(R2PM_PLUGDIR)/r2smt.r2"
	@echo "[r2smt] load with: r2 -i $(R2PM_PLUGDIR)/r2smt.r2 <binary>"

install-prebuilt:
	@if [ -z "$(TRIPLE)" ]; then \
		echo "[r2smt] unsupported triple ($(UNAME_S)/$(UNAME_M)); falling back to source"; \
		exit 1; \
	fi
	@echo "[r2smt] fetching prebuilt $(TARBALL)"
	@mkdir -p $(R2PM_BINDIR)
	@curl -fLsS $(RELEASE_BASE)/$(TARBALL) | tar -xz -C $(R2PM_BINDIR) r2smt

install-from-source:
	cd $(CURDIR)/.. && $(CARGO) build --release -p r2smt-cli
	mkdir -p $(R2PM_BINDIR)
	cp $(CURDIR)/../target/release/r2smt $(R2PM_BINDIR)/r2smt

install-macros:
	mkdir -p $(R2PM_PLUGDIR)
	cp $(CURDIR)/r2smt.r2 $(R2PM_PLUGDIR)/r2smt.r2

uninstall:
	rm -f $(R2PM_BINDIR)/r2smt
	rm -f $(R2PM_PLUGDIR)/r2smt.r2

test:
	$(R2PM_BINDIR)/r2smt version

clean:
	cd $(CURDIR)/.. && $(CARGO) clean -p r2smt-cli
