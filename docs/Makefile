# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = HpBandSter
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

buildapi:
	sphinx-apidoc -fePTMo apidoc ../hpbandster/
	for f in apidoc/*;   \
	do                   \
	  sed -i '/Submodules/d' $$f; \
	  sed -i '/----------/d' $$f; \
	  sed -i 's/    :show-inheritance:/    :show-inheritance:\n    :inherited-members:/g' $$f; \
	done;
	@echo
	@echo "Auto-generation of API documentation finished. " \
	      "The generated files are in 'apidoc/'"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)