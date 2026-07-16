# TravelMind 🧳

TravelMind, **Microsoft Foundry Local** altyapısı kullanılarak geliştirilmiş, **%100 yerel (offline) çalışan** ve gizlilik odaklı bir "Retrieval-Augmented Generation (RAG)" otel öneri asistanıdır. Seçkin bir *Concierge* (otel danışmanı) üslubuna sahip olan bu yapay zeka, CMU TripAdvisor veri seti üzerinden size en uygun otelleri bulur, puanlar ve yorumlar.

Bu proje, **Microsoft Staj Programı (Aşama 1)** kapsamında geliştirilmiştir.

## 🌟 Özellikler
- **Tamamen Çevrimdışı (Offline):** Hiçbir veriniz internete gönderilmez. Hem vektör arama hem de dil modeli (LLM) bilgisayarınızda yerel olarak çalışır.
- **RAG Mimarisi:** Otel verileri parçalara ayrılır (chunking), `all-MiniLM-L6-v2` ile vektörleştirilir (embedding) ve `SQLite` üzerinde vektör araması yapılarak en alakalı sonuçlar çekilir.
- **Elit Concierge Persona:** Dil modeli, "robotik" metinler yerine son derece prestijli, zarif ve profesyonel bir otel danışmanı diliyle yanıt verir.
- **Halüsinasyon Engelleme (Fallback):** Modelin veritabanında olmayan bir bilgi sorulduğunda yalan uydurması katı prompt kurallarıyla (Strict Rules) engellenmiştir. Bilmiyorsa, dürüstçe "Bilmiyorum" der.
- **Dinamik Takip (Follow-up) ve Tercih Daraltma:** Kullanıcının ilk sonuçları beğenmeyip "alternatif öner" veya "daha temizini bul" demesi durumunda, konuşma geçmişini (chat history) koruyarak akıllıca yeni öneriler sunar.

## 🛠️ Kullanılan Teknolojiler
- **Python 3.10+**
- **Microsoft Foundry Local:** Yerel LLM çıkarımı (Inference) için.
- **Phi-4-mini:** Kullanılan yerel küçük boyutlu dil modeli.
- **Sentence-Transformers:** Metinleri sayısal vektörlere dönüştürmek için.
- **SQLite:** Verileri ve vektörleri sunucusuz (serverless) olarak saklamak için.
- **Rich:** Terminalde şık, renkli ve Markdown destekli arayüz sunmak için.

## ⚙️ Kurulum (Setup)

Projeyi bilgisayarınızda çalıştırmak için aşağıdaki adımları sırasıyla izleyin:

1. **Depoyu Klonlayın:**
   ```bash
   git clone <repo_url>
   cd travelmind-rag
   ```

2. **Sanal Ortam (Virtual Environment) Oluşturun ve Aktif Edin:**
   ```bash
   python -m venv .venv
   # Windows için:
   .venv\Scripts\activate
   # macOS/Linux için:
   source .venv/bin/activate
   ```

3. **Gerekli Kütüphaneleri Yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Microsoft Foundry Local'i Başlatın:**
   Foundry uygulamasını açıp modelinizi (örneğin Phi-4-mini) aktif hale getirin. Foundry'nin arka planda çalıştığından ve yerel API portunun açık olduğundan emin olun.

## 🚀 Kullanım (Usage)

Veritabanı halihazırda oluşturulmuş (vektörleştirilmiş) durumdadır. Doğrudan soru-cevap asistanını başlatabilirsiniz:

```bash
python src/cmu_rag_answer.py
```

Uygulama başladığında, konsol üzerinden asistanla sohbet edebilirsiniz. Örnek senaryolar:
- *"Seattle'da iş gezisi için temiz ve sessiz bir otel arıyorum."*
- *"Önerdiğin bu otelleri beğenmedim, başka bir alternatif sunar mısın?"*
- *"Bu otellerden hangisi merkeze daha yakın?"*

## 📁 Proje Yapısı
- `src/cmu_rag_answer.py`: Ana RAG döngüsünü, kullanıcı etkileşimini ve LLM entegrasyonunu içeren merkez dosya.
- `src/cmu_retrieve.py`: Kullanıcı sorgusunu vektörleştirip, veritabanındaki en yakın otel parçalarını bulan (cosine similarity) dosya.
- `src/cmu_recommend_hotels.py`: Vektör aramadan dönen sonuçları puanlayan ve TravelMind formatında biçimlendiren dosya.
- `src/build_cmu_vector_db.py`: Veri setini SQLite veritabanına embedding'ler ile kaydeden veritabanı kurucu script.
- `data/`: CMU TripAdvisor ham veri setini ve oluşturulan SQLite veritabanını (`cmu_travelmind.db`) içerir.
