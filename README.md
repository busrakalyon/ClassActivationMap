# EdgeCAM–Finer-CAM araştırma altyapısı

Bu dizin, özgün `EdgeCAM/` ve `Finer-CAM/` dizinlerine dokunmadan üç deney için ortak,
manifest tabanlı bir altyapı sağlar. Şu an yalnız **Deney 1 (yeniden uygulanan EdgeCAM
baseline)** uygulanmıştır. Finer-CAM ve yöntemler arası füzyon henüz uygulanmamış veya
çalıştırılmamıştır. Gerçek deney sonucu üretilmemiştir.

Gerçek veri hazırlama, güvenli Git güncelleme ve smoke test için kısa yol:
[`RUNBOOK_TR.md`](RUNBOOK_TR.md).

## Bilgi kaynağı ayrımı

### Makale

EdgeCAM PDF sayfa 4, Denklem 2–4 uzamsal konuma özgü ağırlığı tanımlar:

`M_ij = sum_k(w_ijk A_ijk)`, `w_ijk = alpha_ijk ReLU(g_ijk)` ve
`alpha_ijk = g_ijk² / (2g_ijk² + sum_ab(A_ab) g_ijk³)`.

PDF sayfa 5, Denklem 6 yalnız görüntüyü fake sınıflandıran epoch modellerinin CAM'lerini
birleştirir. Aynı sayfadaki Denklem 7–8 ve Algorithm 1, tam 50 SOP maskesinin
`Dil(S)-Ero(S)` kenarlarını üretir, birleşik kenar haritasıyla çarpım toplamını skorlar ve
en yüksek skorlu maskeyi seçer. Sayfa 5 Tablo 1 veri kümelerini; sayfa 5–6 deney
protokolünü; sayfa 6 ArcFace `m=0.5`, `s=50`, 512×512, 100 epoch, AdamW, başlangıç LR
`1e-4`, son LR `1e-8`, batch 16 ve flip/blur/compression artırımlarını verir. Sayfa 8
Tablo 5 maksimum epoch füzyonunun ortalamadan iyi olduğunu, Şekil 3 ve Tablo 8 ise
EfficientNet stage 2 seçimini destekler.

Finer-CAM PDF sayfa 4, Denklem 5–8 hedef–referans logit farkının gradyanını tanımlar.
Bu altyapının `CAMExtractor` arayüzü daha sonra aynı model, checkpoint, katman ve
preprocessing ile yöntemi eklemek içindir; Deney 1 kodu Finer-CAM hesabı içermez.

### İndirilen kaynak kod

`EdgeCAM/hook/EdgeCAM.py`, `torchcam.methods.EdgeCAM` adlı, standart dağıtımda
doğrulanamayan bir sınıfı içe aktarır; sınıfın kaynak kodu bu depoda yoktur. Aynı dosya
EfficientNet-B0'ın 1000 ImageNet logitini `Linear(1000,512)` katmanına verir, her epoch
CAM'i hesaplar, fake tahmin edilmeyen örnek için sıfır harita yazar, CAM'in karesini alarak
kaydeder ve piksel bazında maksimumla birleştirir. `EdgeCAM/models/metrics.py` içindeki
`ArcMarginProduct` varsayılan olarak `s=30`, `m=0.5` kullanır; bu, makaledeki `s=50` ile
uyuşmaz. `EdgeCAM/README.md` sunucu erişiminin kaybedildiğini, kodun kısmi olduğunu ve SOP
adaylarının harici Baidu bağlantısında verildiğini söyler. İndirilen dizinde SOP üretici
kodu/adayları ve yazarın EdgeCAM sınıfı yoktur.

### Bu yeniden uygulamanın kararları

- Denklem 2–4 bağımsız NumPy referansı ve PyTorch uygulaması olarak yazılmıştır. Bu kod
  **yazar implementasyonu değil**, makaleye dayalı yeniden uygulamadır.
- İndirilen koddaki `1000 -> 512` köprüsü varsayılan olarak korunmuştur; makale bu ara
  tasarımı belirtmediğinden `pooled_features` seçeneği yalnız ilerideki ablation içindir.
- ArcFace eğitiminde margin'li logit, validation/test/CAM'de etiketsiz ölçekli kosinüs
  logiti kullanılır. Test etiketi hiçbir çıkarım skoruna verilmez.
- Son CAM ReLU'su ve min–max normalizasyonu makale denklemlerinde açıkça tanımlanmadığı
  için yapılandırılmış yeniden uygulama kararlarıdır. İndirilen koddaki ek CAM karesi
  uygulanmamıştır; bunun ayrı bir ablation olması gerekir.
- Makale periyodik LR düşüşünün kilometre taşlarını açıklamaz. Varsayılan cosine
  `1e-4 -> 1e-8` takvimi bir yeniden uygulama varsayımıdır.
- Makale weight decay'i açıklamaz. İndirilen config `1e-4` yazar, fakat
  `train_mmengine.py` bu değeri AdamW'ye geçirmediği için kod fiilen PyTorch varsayılanını
  kullanır. Yeniden uygulamadaki `1e-4` açıkça kaydedilmiş bir seçimdir.
- Blur olasılığı/kernel dağılımı açıklanmaz; yapılandırmadaki değerler varsayımdır.
- İndirilen `data/dataset.py` gerçekte JPEG sıkıştırma ve yatay çevirmeyi etkinleştirir;
  blur satırı yorumdadır. Ayrıca varsa önceki `latest.jpg` CAM'iyle fake eğitim görüntüsünü
  olasılıksal maskeleyen, makalede açıklanmamış bir yol içerir. Yeniden uygulama bu döngüsel
  CAM-maskeleme davranışını kullanmaz.
- İndirilen `train_mmengine.py` `CosineAnnealingLR(T_max=10)` tanımlar; yeniden uygulamanın
  monotonic 100-epoch cosine takvimi hem bu koddan hem açıklanmayan makale ayrıntısından
  farklıdır ve ablation gerektirir.
- CASIAv2 içindeki %10 kaynak-grubu doğrulama ayrımı makalede yoktur. Test setinde ayar
  yapmamak için eklenmiştir ve literatür protokolünden sapma olarak raporlanmalıdır.
- Makale P-F1'in örnek/mikro/makro birleştirmesini açıkça vermez. Değerlendirici varsayılan
  olarak fake görüntüler üzerinde görüntü-başına makro P-F1/IoU kullanır.
- C-F1, tablolardaki sayılardan doğrulanan `harmonic(I-F1, P-F1)` olarak uygulanmıştır.
- İndirilen `train_mmengine.py:FScore.cal_f1` pozitif sınıf precision/recall F1 döndürür;
  makale I-F1'i sensitivity/specificity ile tarif eder. Değerlendirici ikisini ayrı alanlarda
  tutar ve literatür karşılaştırmasında makale tanımını kullanır.

## Üç deney için mimari

Ortak hat: sürümlenmiş CSV manifest → aynı preprocessing → aynı
`EfficientNetArcFace` → değiştirilebilir CAM çıkarıcı → epoch füzyonu → değiştirilebilir
öneri sağlayıcı → SOP seçimi → ortak metrik/artifact yazıcı.

1. Deney 1: `PaperEdgeCAM` → epoch maksimumu → doğrulanmış 50 hazır SOP varsa seçim.
2. Deney 2 (hazır entegrasyon noktası, uygulanmadı): `CAMExtractor` protokolüne özgün
   hedef–referans logit farkı kullanan Finer-CAM eklenecek.
3. Deney 3 (uygulanmadı): iki çıkarıcı çıktısı aynı uzamsal/provenance sözleşmesiyle ayrı
   bir füzyon modülüne verilecek; lambda yalnız validation setinde seçilecek.

## Dizinler ve önemli dosyalar

- `configs/experiment1_edgecam.toml`: protokol ve açık varsayımlar.
- `data/manifest.py`, `data/casia.py`: provenance, SHA-256, kaynak bağlantılı split ve
  sızıntı denetimi.
- `models/efficientnet_arcface.py`, `models/arcface.py`: model ve etiketsiz inference.
- `models/training.py`, `models/checkpoints.py`: AMP eğitim, her-epoch atomik checkpoint.
- `cams/math_reference.py`, `cams/edgecam.py`: Denklem 2–4.
- `cams/epoch_fusion.py`: fake tahmin kapılı maksimum/ortalama epoch füzyonu.
- `proposals/sop.py`: yalnız Denklem 7–8 seçimi; öneri üretmez.
- `evaluation/metrics.py`: I-F1, AUC, P-F1, IoU ve C-F1.
- `evaluation/thresholds.py`: yalnız validation maskelerinde piksel eşiği seçimi.
- `scripts/`: ortam, manifest, eğitim, CAM, füzyon, SOP, değerlendirme ve artifact CLI'ları.
- `tests/`: veri gerektirmeyen matematik ve sözleşme testleri.

Veri, checkpoint, SOP ve sonuçlar kaynak ağacının dışında tutulur. `paths.example.toml`
yalnız örnektir; sabit yol kodda yoktur.

## Eksik girdiler

- Doğrulanmış CASIAv2/CASIAv1 sürümleri ve ground-truth paketleri.
- CASIA maskelerinin lisans/kaynak URL'si, revision veya arşiv SHA-256 kaydı.
- Yazarın 100 epoch checkpoint'leri (yeniden eğitim yapılacaksa gerekli değildir).
- Yazarın özel `torchcam.methods.EdgeCAM` kaynağı.
- Makaledeki SOP üretici ayarları ve her görüntü için 50 aday SOP artefaktı.

SOP artefaktı yoksa `apply_sop.py` durur. Başka bir segmentasyon yöntemini “özgün SOP”
olarak kullanmaz.

## Ubuntu kurulumu (sudo gerekmez)

Önce proje dizinini ve küçük kaynak kodunu Git/rsync ile aktarın; veri ve checkpoint'leri
ayrı kalıcı dizinlere yerleştirin. Ortamın gerçekten ne olduğunu kaydedin:

```bash
nvidia-smi
python3 --version
uv --version
uv run python scripts/audit_environment.py | tee results/environment.json
```

Mevcut `uv` ortamında çalışan CUDA PyTorch varsa onu koruyun. Yoksa Python 3.12 ortamını
oluşturun ve **o gün** PyTorch'un resmi “Start Locally” seçicisinden Linux/Pip için sürücünün
desteklediği CUDA tekerlek komutunu alın; sistem CUDA toolkit'i veya sudo kurulumu gerekli
değildir. Ardından:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
# Resmi PyTorch seçicisinden alınan torch/torchvision CUDA komutunu burada çalıştırın.
uv pip install -e .
uv pip freeze | tee results/dependencies.freeze.txt
uv run python scripts/audit_environment.py | tee results/environment.json
uv run python -m unittest discover -s tests -v
```

`audit_environment.py` CUDA görünürlüğünü, CUDA runtime'ını, cuDNN'i, GPU adını ve boş/toplam
belleği kaydeder. 16 GB için önce makaledeki batch 16 denenir; OOM olursa batch değişikliği
veya gradient accumulation ayrı protokol sapması olarak kaydedilir, sessizce değiştirilmez.

## Manifest ve veri denetimi

Arşivlerin kaynak URL/revision ve SHA-256 bilgisini deney günlüğünde tutun. CASIAv2 için:

```bash
uv run python scripts/build_casia_manifest.py \
  --dataset casia_v2 \
  --image-root /data/datasets/CASIA2 \
  --mask-root /data/datasets/CASIA2_Groundtruth \
  --output manifests/casia_v2.csv \
  --source-revision "ARCHIVE_NAME_AND_SHA256" \
  --validation-fraction 0.10 --seed 42 --hashes

uv run python scripts/audit_manifest.py \
  --manifest manifests/casia_v2.csv \
  --image-root /data/datasets/CASIA2 \
  --mask-root /data/datasets/CASIA2_Groundtruth --verify-hashes \
  | tee results/casia_v2_audit.json
```

CASIAv1 manifestinde `--dataset casia_v1` kullanılır; bütün satırlar `test` olur. Manifest
üretimi her fake görüntü için tam bir maske eşleşmesi ister, birden çok eşleşmeyi reddeder.
Maskeler `nonzero = manipulated` polaritesine çevrilir. Audit; boyut, değer dağılımı,
boş/tam maske, dosya/hash ve grup-split sızıntısını kontrol eder. Polarite görsel olarak da
en az rastgele 30 örnekte doğrulanmadan eğitime geçilmemelidir.

## Eğitim ve checkpoint

Tam eğitim bu teslimde çalıştırılmamıştır. Veri denetiminden sonra:

```bash
uv run python scripts/train_edgecam_classifier.py \
  --config configs/experiment1_edgecam.toml \
  --manifest manifests/casia_v2.csv \
  --image-root /data/datasets/CASIA2 \
  --output-dir /data/checkpoints/edge_finer/experiment1 \
  --device cuda
```

Her epoch `epoch_001.pt` … `epoch_100.pt`; ayrıca validation I-F1'e göre en iyi checkpoint
yazılır. Checkpoint; model/optimizer/scheduler durumu, epoch, config, validation metrikleri ve
`label_free_scaled_cosine` inference politikasını içerir.

## EdgeCAM, epoch füzyonu, SOP ve değerlendirme

Her checkpoint için aynı CASIAv1 manifestinde CAM üretin:

```bash
for ckpt in /data/checkpoints/edge_finer/experiment1/checkpoints/epoch_*.pt; do
  uv run python scripts/generate_edgecam.py \
    --config configs/experiment1_edgecam.toml \
    --manifest manifests/casia_v1.csv \
    --image-root /data/datasets/CASIA1 \
    --checkpoint "$ckpt" \
    --output-dir /data/results/edge_finer/experiment1/raw_cams \
    --split test --device cuda --batch-size 4
done

uv run python scripts/fuse_epoch_cams.py \
  --manifest manifests/casia_v1.csv \
  --cam-root /data/results/edge_finer/experiment1/raw_cams \
  --output-dir /data/results/edge_finer/experiment1/fused \
  --split test --method max --threshold 0.5
```

Ham CAM 512×512 üretilir; epoch füzyonundan sonra bilinear olarak özgün görüntü boyutuna
taşınır, maske eşiklemesi bundan sonra yapılır. Her epoch `.npz` dosyası logit, tahmin ve
füzyon geçerlilik bayrağını korur.

Piksel eşiğini CASIAv2 `val` CAM'leri üzerinde seçin; araç `test` splitini açıkça reddeder:

```bash
uv run python scripts/select_pixel_threshold.py \
  --manifest manifests/casia_v2.csv \
  --mask-root /data/datasets/CASIA2_Groundtruth \
  --fused-cam-dir /data/results/edge_finer/experiment1/validation_fused \
  --output /data/results/edge_finer/experiment1/selected_threshold.json
```

Kaydedilen eşik daha sonra CASIAv1 ve diğer test kümelerindeki
`fuse_epoch_cams.py --threshold` argümanına aynen verilmelidir; test sonuçlarına bakarak
değiştirilmemelidir.

Doğrulanmış 50-aday SOP varsa:

```bash
uv run python scripts/apply_sop.py \
  --manifest manifests/casia_v1.csv \
  --fused-cam-dir /data/results/edge_finer/experiment1/fused \
  --proposal-dir /data/datasets/edgecam_sop/casia_v1 \
  --output-dir /data/results/edge_finer/experiment1/post_sop
```

SOP öncesi değerlendirme; görüntü metrikleri için makale protokolündeki epoch 100 CSV'si:

```bash
uv run python scripts/evaluate.py \
  --manifest manifests/casia_v1.csv \
  --mask-root /data/datasets/CASIA1_Groundtruth \
  --prediction-dir /data/results/edge_finer/experiment1/fused \
  --prediction-suffix __pre_sop.npy \
  --image-predictions /data/results/edge_finer/experiment1/raw_cams/epoch_100/image_predictions.csv \
  --output-dir /data/results/edge_finer/experiment1/evaluation/pre_sop \
  --split test --pixel-scope fake
```

SOP sonrası aynı komut `--prediction-dir .../post_sop` ve
`--prediction-suffix __post_sop.npy` ile çalışır. Görselleri ve örnek bazlı metrikleri dışa
aktarmak için `scripts/export_artifacts.py` kullanılır.

## Deney kapıları

Deney 2'ye geçmeden önce zorunlu kapılar:

1. CASIAv2/CASIAv1 sürüm, maske kaynağı, eşleşme ve polarite denetimi temiz olmalı.
2. Ubuntu'da tüm testler, PyTorch paritesi ve ArcFace testi dahil geçmeli.
3. Label-free çıkarım sınıflandırma metrikleri ve epoch 100 checkpoint'i doğrulanmalı.
4. En az CASIAv1'de ham EdgeCAM ve epoch-füzyon sonuçları alınmalı.
5. SOP artefaktı bulunursa kaynağı/hash'i kaydedilmeli; bulunmazsa baseline “SOP öncesi” ve
   “SOP eksik” olarak raporlanmalı, uydurma sonuç üretilmemeli.
6. Yeniden uygulama sonuçları makale Tablo 3–5 ve 8'den kopyalanmış literatür değerleriyle
   ayrı satırlarda karşılaştırılmalı.
