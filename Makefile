# r2SMT build and radare2 integration targets.
#
#   make                 Build the release CLI.
#   make plugin          Build the native radare2 core plugin.
#   make user-install    Install the CLI, core plugin, and compatibility aliases.
#   make symstall        Symlink the CLI and integration files for development.

CARGO ?= cargo
CC ?= cc
PKG_CONFIG ?= pkg-config
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
PLUGIN_SOURCE := r2pm/core_r2smt.c
R2_LIBEXT ?= $(shell $(R2) -H R2_LIBEXT 2>/dev/null || printf so)
R2_ABIVERSION ?= $(shell $(R2) -H R2_ABIVERSION 2>/dev/null || printf unknown)
PLUGIN_NAME := core_r2smt.$(R2_LIBEXT)
PLUGIN_TARGET := target/r2-plugin/$(R2_ABIVERSION)/$(PLUGIN_NAME)
R2_PLUGIN_CFLAGS := $(shell $(PKG_CONFIG) --cflags r_core 2>/dev/null)
R2_PLUGIN_LIBS := $(shell $(PKG_CONFIG) --libs r_core 2>/dev/null)
PLUGIN_WARNINGS ?= -Wall -Wextra

ifeq ($(shell uname -s),Darwin)
PLUGIN_SHARED_FLAG := -dynamiclib
else
PLUGIN_SHARED_FLAG := -shared
endif

.DEFAULT_GOAL := all

all:
	$(CARGO) build --release -p $(PACKAGE)

check-r2-dev:
	@$(PKG_CONFIG) --exists r_core || { \
		echo "error: radare2 development files were not found (pkg-config r_core)" >&2; \
		exit 1; \
	}

$(PLUGIN_TARGET): $(PLUGIN_SOURCE) | check-r2-dev
	mkdir -p "$(dir $(PLUGIN_TARGET))"
	$(CC) $(CPPFLAGS) $(CFLAGS) $(PLUGIN_WARNINGS) -fPIC $(PLUGIN_SHARED_FLAG) \
		$(R2_PLUGIN_CFLAGS) -o "$@" "$<" $(LDFLAGS) $(R2_PLUGIN_LIBS)

plugin: $(PLUGIN_TARGET)

plugin-install: plugin
	mkdir -p "$(DESTDIR)$(PLUGDIR)"
	install -m 755 "$(PLUGIN_TARGET)" "$(DESTDIR)$(PLUGDIR)/$(PLUGIN_NAME)"
	install -m 644 "$(MACRO)" "$(DESTDIR)$(PLUGDIR)/$(notdir $(MACRO))"

install: all plugin
	mkdir -p "$(DESTDIR)$(BINDIR)" "$(DESTDIR)$(PLUGDIR)"
	install -m 755 "$(TARGET)" "$(DESTDIR)$(BINDIR)/$(BIN)"
	install -m 755 "$(PLUGIN_TARGET)" "$(DESTDIR)$(PLUGDIR)/$(PLUGIN_NAME)"
	install -m 644 "$(MACRO)" "$(DESTDIR)$(PLUGDIR)/$(notdir $(MACRO))"

uninstall:
	rm -f "$(DESTDIR)$(BINDIR)/$(BIN)"
	rm -f "$(DESTDIR)$(PLUGDIR)/$(PLUGIN_NAME)"
	rm -f "$(DESTDIR)$(PLUGDIR)/$(notdir $(MACRO))"

# Use r2pm's bin directory for the CLI and r2's auto-loaded per-user
# plugin directory for the native bridge and compatibility aliases.
user-install:
	$(MAKE) install BINDIR="$(R2PM_BINDIR)" PLUGDIR="$(R2_USER_PLUGINS)"

user-uninstall:
	$(MAKE) uninstall BINDIR="$(R2PM_BINDIR)" PLUGDIR="$(R2_USER_PLUGINS)"

# Keep a checkout live while developing. Re-run `make plugin` after C changes
# and `make` after Rust changes.
symstall: all plugin
	mkdir -p "$(R2PM_BINDIR)" "$(R2_USER_PLUGINS)"
	ln -sfn "$(abspath $(TARGET))" "$(R2PM_BINDIR)/$(BIN)"
	ln -sfn "$(abspath $(PLUGIN_TARGET))" "$(R2_USER_PLUGINS)/$(PLUGIN_NAME)"
	ln -sfn "$(abspath $(MACRO))" "$(R2_USER_PLUGINS)/$(notdir $(MACRO))"

clean:
	$(CARGO) clean -p $(PACKAGE)
	rm -f "$(PLUGIN_TARGET)"

mrproper: clean

test:
	$(CARGO) test --workspace

check:
	$(CARGO) check --workspace --all-targets
	$(MAKE) plugin

plugin-smoke: plugin
	./scripts/r2-plugin-smoke.sh "$(PLUGIN_TARGET)"

fmt format:
	$(CARGO) fmt --all

lint:
	$(CARGO) clippy --workspace --all-targets --all-features -- -D warnings

.PHONY: all check-r2-dev plugin plugin-install install uninstall user-install \
	user-uninstall symstall clean mrproper test check plugin-smoke fmt format lint
