
package OnnxSTT_h;
use strict; use Cwd 'realpath';  use File::Basename;  use lib dirname(__FILE__);  use SPL::Operator::Instance::OperatorInstance; use SPL::Operator::Instance::Annotation; use SPL::Operator::Instance::Context; use SPL::Operator::Instance::Expression; use SPL::Operator::Instance::ExpressionTree; use SPL::Operator::Instance::ExpressionTreeEvaluator; use SPL::Operator::Instance::ExpressionTreeVisitor; use SPL::Operator::Instance::ExpressionTreeCppGenVisitor; use SPL::Operator::Instance::InputAttribute; use SPL::Operator::Instance::InputPort; use SPL::Operator::Instance::OutputAttribute; use SPL::Operator::Instance::OutputPort; use SPL::Operator::Instance::Parameter; use SPL::Operator::Instance::StateVariable; use SPL::Operator::Instance::TupleValue; use SPL::Operator::Instance::Window; 
sub main::generate($$) {
   my ($xml, $signature) = @_;  
   print "// $$signature\n";
   my $model = SPL::Operator::Instance::OperatorInstance->new($$xml);
   unshift @INC, dirname ($model->getContext()->getOperatorDirectory()) . "/../impl/nl/include";
   $SPL::CodeGenHelper::verboseMode = $model->getContext()->isVerboseModeOn();
   print '// Additional includes for OnnxSTT operator', "\n";
   print '#include "../../../impl/include/OnnxSTTInterface.hpp"', "\n";
   print '#include "../../../impl/include/StreamingBuffer.hpp"', "\n";
   print '#include <memory>', "\n";
   print "\n";
   SPL::CodeGen::headerPrologue($model);
   print "\n";
   print "\n";
   print 'class MY_OPERATOR : public MY_BASE_OPERATOR ', "\n";
   print '{', "\n";
   print 'public:', "\n";
   print '    // Constructor', "\n";
   print '    MY_OPERATOR();', "\n";
   print '    ', "\n";
   print '    // Destructor', "\n";
   print '    virtual ~MY_OPERATOR();', "\n";
   print '    ', "\n";
   print '    // Tuple processing for non-mutating ports', "\n";
   print '    void process(Tuple const & tuple, uint32_t port);', "\n";
   print '    ', "\n";
   print '    // Punctuation processing', "\n";
   print '    void process(Punctuation const & punct, uint32_t port);', "\n";
   print '    ', "\n";
   print 'private:', "\n";
   print '    // Mutex for thread safety', "\n";
   print '    SPL::Mutex _mutex;', "\n";
   print '    ', "\n";
   print '    // Configuration', "\n";
   print '    onnx_stt::OnnxSTTInterface::Config config_;', "\n";
   print '    ', "\n";
   print '    // ONNX-based implementation', "\n";
   print '    std::unique_ptr<onnx_stt::OnnxSTTInterface> onnx_impl_;', "\n";
   print '    ', "\n";
   print '    // State tracking', "\n";
   print '    bool initialized_;', "\n";
   print '    uint64_t audio_timestamp_ms_;', "\n";
   print '    uint64_t total_samples_processed_;', "\n";
   print '    ', "\n";
   print '    // Streaming support', "\n";
   print '    bool streaming_mode_;', "\n";
   print '    int32_t chunk_overlap_ms_;', "\n";
   print '    std::unique_ptr<onnx_stt::StreamingBuffer> streaming_buffer_;', "\n";
   print '    ', "\n";
   print '    // Helper methods', "\n";
   print '    void initialize();', "\n";
   print '    void processAudioData(const SPL::blob& audio_blob);', "\n";
   print '    void processStreamingAudio(const float* samples, size_t num_samples);', "\n";
   print '    void submitResult(const onnx_stt::OnnxSTTInterface::TranscriptionResult& result);', "\n";
   print '};', "\n";
   print "\n";
   SPL::CodeGen::headerEpilogue($model);
   print "\n";
   CORE::exit $SPL::CodeGen::USER_ERROR if ($SPL::CodeGen::sawError);
}
1;
