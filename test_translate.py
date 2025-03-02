#pip install sonar-space==0.2.1
#pip install fairseq2==0.2.1 
from sonar.inference_pipelines.text import TextToTextModelPipeline
print("initializing the Text-to-Text model pipeline...")
t2t_model = TextToTextModelPipeline(encoder="text_sonar_basic_encoder",
                                    decoder="text_sonar_basic_decoder",
                                    tokenizer="text_sonar_basic_encoder")  # tokenizer is attached to both encoder and decoder cards
print("Text-to-Text model pipeline initialized successfully.")
sentences = ['I can embed the sentences into vectorial space.']
print(type(sentences[0]))
print(t2t_model.predict(sentences, source_lang="eng_Latn", target_lang="arb_Arab"))