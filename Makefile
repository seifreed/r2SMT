# r2SMT build and radare2 integration targets.
#
#   make                 Build the release CLI.
#   make user-install    Install the CLI and r2 macros for this user.
#   make symstall        Symlink the built CLI and macros for development.

CARGO ?= cargo
R2 ?= r2
R2PM ?= r2pm

PREFIX ?= /usr/local
BINDIR ?= $(PREFIX)/bin
PLUGDIR ?= $(shell $(R2) -H R2_LIBR_PLUGINS 2>/dev/null || printf '%s/lib/radare2/plugins' '$(PREFIX)')

R2PM_BINDIR ?= $(shell $(R2PM) -H R2PM_BINDIR 2>/dev/null || printf '%s/.local/share/radare2/prefix/bin' '$(HOME)')
R2_USER_PLUGINS ?= $(shell $(R2) -H R2_USER_PLUGINS 2>/dev/null || printf '%s/.local/share/radare2/plugins' '$(HOME)')

PACKAGE := r2smt-cli
BIN := r2smt
TARGET := target/release/$(BIN)
MACRO := r2pm/r2smt.r2

.DEFAULT_GOAL := all

all:
	$(CARGO) build --release -p $(PACKAGE)

install: all
	mkdir -p "$(DESTDIR)$(BINDIR)" "$(DESTDIR)$(PLUGDIR)"
	install -m 755 "$(TARGET)" "$(DESTDIR)$(BINDIR)/$(BIN)"
	install -m 644 "$(MACRO)" "$(DESTDIR)$(PLUGDIR)/$(notdir $(MACRO))"

uninstall:
	rm -f "$(DESTDIR)$(BINDIR)/$(BIN)"
	rm -f "$(DESTDIR)$(PLUGDIR)/$(notdir $(MACRO))"

# Use r2pm's bin directory so the macro's `!r2smt ...` shell-outs find
# the executable, and r2's per-user plugin directory for the macros.
user-install:
	$(MAKE) install BINDIR="$(R2PM_BINDIR)" PLUGDIR="$(R2_USER_PLUGINS)"

user-uninstall:
	$(MAKE) uninstall BINDIR="$(R2PM_BINDIR)" PLUGDIR="$(R2_USER_PLUGINS)"

# Keep a checkout live while developing.  Re-run `make` after Rust changes.
symstall: all
	mkdir -p "$(R2PM_BINDIR)" "$(R2_USER_PLUGINS)"
	ln -sfn "$(abspath $(TARGET))" "$(R2PM_BINDIR)/$(BIN)"
	ln -sfn "$(abspath $(MACRO))" "$(R2_USER_PLUGINS)/$(notdir $(MACRO))"

clean:
	$(CARGO) clean -p $(PACKAGE)

mrproper: clean

test:
	$(CARGO) test --workspace

check:
	$(CARGO) check --workspace --all-targets

fmt format:
	$(CARGO) fmt --all

lint:
	$(CARGO) clippy --workspace --all-targets --all-features -- -D warnings

.PHONY: all install uninstall user-install user-uninstall symstall clean mrproper \
	test check fmt format lint
