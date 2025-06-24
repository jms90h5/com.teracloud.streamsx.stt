# Makefile for NeMo CTC Speech-to-Text Samples
# Builds sample applications using the working NeMo CTC implementation

STREAMS_INSTALL ?= $(HOME)/teracloud/streams/7.2.0.0
TOOLKIT_PATH = ..

# Samples to build
SAMPLES = BasicNeMoDemo NeMoCTCRealtime NeMoFileTranscription

# Default target
all: $(SAMPLES)

# Ensure toolkit is built first
check-toolkit:
	@echo "Checking if toolkit is built..."
	@if [ ! -f "$(TOOLKIT_PATH)/impl/lib/libnemo_stt.so" ]; then \
		echo "Building toolkit first..."; \
		$(MAKE) -C $(TOOLKIT_PATH); \
	fi

# Build individual samples
BasicNeMoDemo: check-toolkit
	@echo "Building BasicNeMoDemo..."
	@mkdir -p output/BasicNeMoDemo
	sc -a -t $(TOOLKIT_PATH) -M BasicNeMoDemo --output-directory output/BasicNeMoDemo
	@echo "✅ BasicNeMoDemo built successfully"

NeMoCTCRealtime: check-toolkit
	@echo "Building NeMoCTCRealtime..."
	@mkdir -p output/NeMoCTCRealtime
	sc -a -t $(TOOLKIT_PATH) -M NeMoCTCRealtime --output-directory output/NeMoCTCRealtime
	@echo "✅ NeMoCTCRealtime built successfully"

NeMoFileTranscription: check-toolkit
	@echo "Building NeMoFileTranscription..."
	@mkdir -p output/NeMoFileTranscription
	sc -a -t $(TOOLKIT_PATH) -M NeMoFileTranscription --output-directory output/NeMoFileTranscription
	@echo "✅ NeMoFileTranscription built successfully"

# Clean build artifacts
clean:
	rm -rf output/
	@echo "✅ Cleaned build artifacts"

# Test samples (requires models and test data)
test: all
	@echo "Testing BasicNeMoDemo..."
	@if [ -f "$(TOOLKIT_PATH)/models/fastconformer_ctc_export/model.onnx" ] && \
	   [ -f "$(TOOLKIT_PATH)/test_data/audio/librispeech-1995-1837-0001.wav" ]; then \
		streamtool submitjob output/BasicNeMoDemo/BasicNeMoDemo.sab; \
		echo "✅ BasicNeMoDemo test submitted"; \
	else \
		echo "❌ Missing models or test data. Run 'make setup-models' first."; \
	fi

# Test samples with real-time throttling
test-throttled: all
	@echo "Testing BasicNeMoDemo with real-time throttling..."
	@if [ -f "$(TOOLKIT_PATH)/models/fastconformer_ctc_export/model.onnx" ] && \
	   [ -f "$(TOOLKIT_PATH)/test_data/audio/librispeech-1995-1837-0001.wav" ]; then \
		streamtool submitjob output/BasicNeMoDemo/BasicNeMoDemo.sab -P enableThrottling=true; \
		echo "✅ BasicNeMoDemo throttled test submitted"; \
	else \
		echo "❌ Missing models or test data. Run 'make setup-models' first."; \
	fi

# Setup required models and test data
setup-models:
	@echo "Setting up models and test data..."
	@if [ ! -d "$(TOOLKIT_PATH)/models/fastconformer_ctc_export" ]; then \
		echo "❌ NeMo CTC model not found. Please run model export first:"; \
		echo "   cd $(TOOLKIT_PATH) && python export_model_ctc_patched.py"; \
		exit 1; \
	fi
	@if [ ! -f "$(TOOLKIT_PATH)/test_data/audio/librispeech-1995-1837-0001.wav" ]; then \
		echo "❌ Test audio files not found. Please ensure test data is available."; \
		exit 1; \
	fi
	@echo "✅ Models and test data are ready"

# Show sample status
status:
	@echo "=== NeMo CTC Samples Status ==="
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
	@echo "Required models:"
	@if [ -f "$(TOOLKIT_PATH)/models/fastconformer_ctc_export/model.onnx" ]; then \
		echo "  ✅ NeMo CTC model ($(shell stat -c%s $(TOOLKIT_PATH)/models/fastconformer_ctc_export/model.onnx 2>/dev/null | numfmt --to=iec)B)"; \
	else \
		echo "  ❌ NeMo CTC model (missing)"; \
	fi
	@if [ -f "$(TOOLKIT_PATH)/models/fastconformer_ctc_export/tokens.txt" ]; then \
		echo "  ✅ Vocabulary file ($(shell wc -l < $(TOOLKIT_PATH)/models/fastconformer_ctc_export/tokens.txt 2>/dev/null) tokens)"; \
	else \
		echo "  ❌ Vocabulary file (missing)"; \
	fi
	@echo ""
	@echo "Test data:"
	@if [ -f "$(TOOLKIT_PATH)/test_data/audio/librispeech-1995-1837-0001.wav" ]; then \
		echo "  ✅ LibriSpeech test file"; \
	else \
		echo "  ❌ LibriSpeech test file (missing)"; \
	fi
	@if [ -f "$(TOOLKIT_PATH)/test_data/audio/11-ibm-culture-2min-16k.wav" ]; then \
		echo "  ✅ IBM Culture test file"; \
	else \
		echo "  ❌ IBM Culture test file (missing)"; \
	fi

# Help target
help:
	@echo "NeMo CTC Speech-to-Text Samples"
	@echo "================================"
	@echo ""
	@echo "Available targets:"
	@echo "  all              - Build all samples"
	@echo "  BasicNeMoDemo    - Build basic demo sample"
	@echo "  NeMoCTCRealtime  - Build real-time processing sample"
	@echo "  NeMoFileTranscription - Build file transcription sample"
	@echo "  clean            - Remove build artifacts"
	@echo "  test             - Test samples (requires models)"
	@echo "  test-throttled   - Test samples with real-time throttling"
	@echo "  setup-models     - Check model and test data availability"
	@echo "  status           - Show build and dependency status"
	@echo "  help             - Show this help"
	@echo ""
	@echo "Prerequisites:"
	@echo "  1. Source Streams environment: source $(STREAMS_INSTALL)/bin/streamsprofile.sh"
	@echo "  2. Build toolkit: make -C $(TOOLKIT_PATH)"
	@echo "  3. Export NeMo model: cd $(TOOLKIT_PATH) && python export_model_ctc_patched.py"
	@echo ""
	@echo "Quick start:"
	@echo "  make status      # Check dependencies"
	@echo "  make all         # Build all samples"
	@echo "  make test        # Test basic sample"
	@echo "  make test-throttled # Test with real-time throttling"

.PHONY: all clean test test-throttled setup-models status help check-toolkit $(SAMPLES)