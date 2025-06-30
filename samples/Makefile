# Makefile for STT Toolkit Samples
# 
# Note: Current status shows samples may not compile due to ONNX Runtime header conflicts.
# See doc/ARCHITECTURE.md for details on known issues.

STREAMS_INSTALL ?= $(HOME)/teracloud/streams/7.2.0.1
TOOLKIT_PATH = ..

# Available samples
SAMPLES = BasicSTTExample SimpleTest WorkingNeMoRealtime NeMoCTCSample

# Default target
all: check-environment $(SAMPLES)

# Check if environment is properly set up
check-environment:
	@echo "Checking build environment..."
	@if [ -z "$(STREAMS_INSTALL)" ]; then \
		echo "❌ STREAMS_INSTALL not set"; \
		exit 1; \
	fi
	@if [ ! -f "$(TOOLKIT_PATH)/toolkit.xml" ]; then \
		echo "❌ Toolkit not indexed. Run: cd $(TOOLKIT_PATH) && spl-make-toolkit -i ."; \
		exit 1; \
	fi
	@echo "✅ Environment checks passed"

# Build basic example
BasicSTTExample: check-environment
	@echo "Building BasicSTTExample..."
	@mkdir -p output/BasicSTTExample
	sc -a -t $(TOOLKIT_PATH) -M BasicSTTExample \
		--output-directory output/BasicSTTExample \
		BasicSTTExample.spl
	@echo "✅ BasicSTTExample build attempted"

# Clean build artifacts
clean:
	rm -rf output/
	@echo "✅ Cleaned build artifacts"

# Check sample status
status:
	@echo "=== STT Toolkit Samples Status ==="
	@echo "Toolkit path: $(TOOLKIT_PATH)"
	@echo "Streams install: $(STREAMS_INSTALL)"
	@echo ""
	@echo "Available samples:"
	@for sample in $(SAMPLES); do \
		if [ -f "output/$$sample/$$sample.sab" ]; then \
			echo "  ✅ $$sample (built)"; \
		else \
			echo "  ⏸️  $$sample (not built)"; \
		fi; \
	done
	@echo ""
	@echo "Known Issues:"
	@echo "  ⚠️  ONNX Runtime header conflicts may prevent compilation"
	@echo "  ⚠️  See doc/ARCHITECTURE.md for troubleshooting"

# Help
help:
	@echo "STT Toolkit Samples"
	@echo "==================="
	@echo ""
	@echo "Available targets:"
	@echo "  all              - Build all samples"
	@echo "  BasicSTTExample  - Build basic example"
	@echo "  clean            - Remove build artifacts"
	@echo "  status           - Show build status"
	@echo "  help             - Show this help"
	@echo ""
	@echo "Prerequisites:"
	@echo "  1. Source Streams environment"
	@echo "  2. Index toolkit: cd $(TOOLKIT_PATH) && spl-make-toolkit -i ."
	@echo "  3. Set up models (see toolkit documentation)"
	@echo ""
	@echo "Known Issues:"
	@echo "  - ONNX Runtime header conflicts may prevent compilation"
	@echo "  - See doc/ARCHITECTURE.md for current status and fixes"

.PHONY: all clean status help check-environment $(SAMPLES)