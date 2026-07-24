# Model Decision

TravelMind RAG projesinde ana cevap üretme modeli olarak Phi-4-mini-instruct kullanılacaktır.

Test edilen modeller:

1. Phi-4-mini-instruct
   - Seyahat/otel yorumları alanında temiz ve doğrudan cevap üretmiştir.
   - Ana local LLM olarak kullanılacaktır.

2. qwen3-0.6b
   - Local olarak başarıyla çalıştırılmıştır.
   - Ancak cevap üretmeden önce "Thinking..." çıktısı verdiği için ana kullanıcı arayüzünde kullanılmayacaktır.
   - Sadece alternatif/test modeli olarak değerlendirilecektir.

Son karar:
Ana model: Phi-4-mini-instruct
Alternatif model: qwen3-0.6b
