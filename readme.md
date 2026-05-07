-> VireoFood172 - Training a model from scratch for the multi-ingredient recognition task. 
** Dataset Link: https://fvl.fudan.edu.cn/dataset/vireofood172/list.htm -- Form doldurulup gönderilecek.

[Kullanıcı Girdileri] ---> (Fotoğraf) + (Metin/Bağlam) + (Opsiyonel Barkod)
       |
       v
+====================================================================+
|                     ÇOKLU AJAN İŞLEME KATMANI                      |
|                                                                    |
|  👁️ 1. VISION AGENT (İkili Alt-Model Stratejisi)                    |
|      A) YOLOv8 (İçerik ve Alan Tespiti)                            |
|         - Oto-etiketlenmiş (Grounding DINO) veri setiyle eğitilir. |
|         - Çıktı: Malzemeler ve ekrandaki piksel kaplama oranları.  |
|      B) ResNet50 (Kütle Regresyonu)                                |
|         - Nutrition5k (3GB + Augmentation) ile eğitilir.           |
|         - Çıktı: Tabağın Toplam Gramajı.                           |
|      * Dahili Matematik: Toplam Gramaj, piksel alanına göre bölünür|
|        (Örn: 350g Toplam -> %60 Pirinç [210g], %40 Tavuk [140g]).  |
|                                                                    |
|  📝 2. TEXT AGENT (Bağlam ve Doğrulama)                             |
|      - Kullanıcı metnini işler (NLP).                              |
|      - Çıktı: Gizli içerikler (Örn: "Zeytinyağlı", "Suyunu süzdüm")|
|                                                                    |
|  🛒 3. BARCODE AGENT (Deterministik Gerçeklik)                      |
|      - OpenFoodFacts API ile konuşur.                              |
|      - Çıktı: Spesifik ürünün 100g başı kesin makroları.           |
+====================================================================+
       |
       | (Tüm ajanlar verilerini standart bir JSON Pydantic formatında basar)
       v
+====================================================================+
|                    LLM FUSION AGENT (Ana Beyin)                    |
|                                                                    |
|  - Çatışma Çözümü: Görüntü "Dana" der, metin "Soya Eti" derse      |
|    metne (kullanıcıya) güvenir, porsiyonu görüntüden alır.         |
|  - Mutfak Mantığı: Çiğ/Pişmiş ağırlık dönüşümlerini yapar.         |
|  - Nihai Hesaplama: Tüm verileri matematiksel olarak sentezler.    |
+====================================================================+
       |
       v
[Nihai Çıktı: Yemek Adı, Kesin Makrolar ve "Hesaplama Gerekçesi" Açıklaması]

Faz 1: Veri Üretimi (Araştırma Aşaması)

VireoFood-172 veya benzeri bir veri setini indirmek.

Açık kaynaklı bir sıfır-atış (zero-shot) modeli (SAM veya Grounding DINO) kullanarak bu fotoğraflardaki malzemelere otomatik bounding box çizdirmek.

Bu koordinatları alıp kendimize ait, yepyeni bir YOLO veri seti oluşturmak.

Faz 2: Ajanların Modellerini Eğitme (Colab Aşaması)

Oluşturduğumuz yeni veri setiyle YOLOv8 tespit modelini eğitmek.

Nutrition5k verisi ve bol Data Augmentation ile ResNet50 regresyon modelini (sadece ağırlık tahmini için) eğitmek.

Faz 3: Arka Uç (Backend) Entegrasyonu

Daha önce iskeletini çıkardığımız modüler klasör yapısını (FastAPI) kurmak.

Pydantic ile tüm ajanların aynı dili (AgentResponse JSON şeması) konuşmasını sağlamak.

Ajanları API uç noktalarına (endpoints) bağlamak.

Faz 4: LLM Füzyon ve Test

Hazırladığımız "System Prompt" ile LLM'i sisteme entegre etmek.

Sisteme gerçek fotoğraflar ve metinler atarak ajanların uyumunu ve LLM'in mantıksal çıkarım (reasoning_summary) yeteneğini test etmek.