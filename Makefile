# Main Makefile for TeraCloud Streams Speech-to-Text Toolkit

.PHONY: all clean

# Build SPL toolkit index
all:
	@echo "Building SPL toolkit index..."
	$(STREAMS_INSTALL)/bin/spl-make-toolkit -i .

clean:
	@echo "Cleaning toolkit..."
	rm -f toolkit.xml

# Build everything (implementation + toolkit)
build-all: 
	@echo "Building implementation library..."
	$(MAKE) -C impl
	@echo "Building SPL toolkit..."
	$(MAKE) all

# Clean everything
clean-all:
	@echo "Cleaning implementation..."
	$(MAKE) -C impl clean
	@echo "Cleaning toolkit..."
	$(MAKE) clean

# Test cache-aware NeMo model
test-nemo-cache-aware: build-all
	@echo "Building and running NeMo Cache-Aware Streaming test..."
	g++ -std=c++14 -O2 -I./impl/include -I./deps/onnxruntime/include \
		test_nemo_cache_aware.cpp impl/lib/libs2t_impl.so \
		-L./deps/onnxruntime/lib -lonnxruntime \
		-Wl,-rpath,'$$ORIGIN/impl/lib' -Wl,-rpath,'$$ORIGIN/deps/onnxruntime/lib' \
		-o test_nemo_cache_aware
	./test_nemo_cache_aware