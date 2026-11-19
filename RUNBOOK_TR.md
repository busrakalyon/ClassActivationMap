# EdgeCAM Deney 1 — kısa Ubuntu runbook

## 1. Veri kaynakları

| Veri | Önerilen kaynak | Durum |
|---|---|---|
| CASIAv2 görüntü + düzeltilmiş maske | `SunnyHaze/CASIA2.0-Corrected-Groundtruth`, Drive ID `1JN93ts7VxJ4bbJL9TlE9uxM3ZeYuCN9M` | Görüntü ve maskeyi birlikte verir; pratikte en tutarlı başlangıç. Yazarların birebir sürümü olduğu kanıtlanmış değildir. |
| CASIAv2 topluluk maskesi | `namtpham/casia2groundtruth`, `CASIA2.0_Groundtruth.zip` | Pham vd. maskeleri; eski paketlerde isim ve maske gürültüsü raporlanmıştır. |
| CASIAv1 değiştirilmiş görüntüler | `namtpham/casia1groundtruth`, Drive ID `14f3jU2VsxTYopgSE1Vvv4hMlFvpAKHUY` | 800 gerçek/920 sahte sürüme en yakın erişilebilir paket. |
| CASIAv1 maskeleri | `namtpham/casia1groundtruth`, `CASIA 1.0 groundtruth.zip` | Topluluk maskesi; görüntülerle audit zorunlu. |

Önemli: yaygın CASIAv2 sürümü 5.123 sahte görüntü bildirirken EdgeCAM Tablo 1, 5.063
bildirir. Yazarların dosya listesi yayımlanmadığından tam aynı split doğrulanamaz. Kullanılan
ZIP'in SHA-256'sı ve manifest sayıları rapora yazılmalıdır.

MVSS-Net ayrıca bazı CASIAv1 gerçek görüntülerinin CASIAv2 içinde de bulunduğunu belgeler.
EdgeCAM literatür karşılaştırması için özgün CASIAv1 sonucu aynen tutulmalı; sızıntısız
ikincil analizde CASIAv1plus veya SHA-256 ile CASIAv2 çakışmaları çıkarılmış CASIAv1 ayrı
satırda raporlanmalıdır. Bu iki protokolün sonuçları birbirinin yerine kullanılmamalıdır.

Ubuntu'da, proje dışında:

```bash
mkdir -p /data/edgecam/downloads /data/edgecam/datasets

uvx gdown 1JN93ts7VxJ4bbJL9TlE9uxM3ZeYuCN9M \
  -O /data/edgecam/downloads/CASIA2_corrected.zip
uv run python -m zipfile -e /data/edgecam/downloads/CASIA2_corrected.zip \
  /data/edgecam/datasets/CASIA2_corrected

uvx gdown 14f3jU2VsxTYopgSE1Vvv4hMlFvpAKHUY \
  -O /data/edgecam/downloads/CASIA1_modified.zip
uv run python -m zipfile -e /data/edgecam/downloads/CASIA1_modified.zip \
  /data/edgecam/datasets/CASIA1_modified

uv run python scripts/download_file.py \
  "https://raw.githubusercontent.com/namtpham/casia1groundtruth/master/CASIA%201.0%20groundtruth.zip" \
  /data/edgecam/downloads/CASIA1_groundtruth.zip
uv run python -m zipfile -e /data/edgecam/downloads/CASIA1_groundtruth.zip \
  /data/edgecam/datasets/CASIA1_groundtruth

sha256sum /data/edgecam/downloads/*.zip
```

CASIA2 corrected paket zaten `Au/Tp/Gt` içerir. Çıkan gerçek kökleri kontrol edip aşağıdaki
komutlardaki yolları gerektiği kadar düzeltin.

## 2. SOP

Bu çalışmada yazarların precomputed SOP paketi kullanılmaz. Tek proposal üreticisi resmî
`jponttuset/mcg` deposunun `pre-trained` sürümüdür ve sonuçlar açıkça
`MCG-reconstructed SOP` olarak raporlanır. Baidu indirmesi ve yazar e-postası iş akışından
çıkarılmıştır.

Ubuntu smoke testinde MATLAB R2026b, gerekli toolbox'lar, `install.m` ve
`im2mcg(I,'accurate',0)` doğrulanmıştır. Resmî MCG çıktısı zaten azalan proposal skoruna göre
sıralıdır. İlk 50 maske, bütün binlerce maskeyi materialize etmek yerine
`ismember(proposals.superpixels, proposals.labels{i})` ile çıkarılır.

Ortak cache sözleşmesi kayıpsız sıkıştırılmış NPZ'dir:

- `masks`: `[H,W,50]`, binary `uint8`;
- `scores`: resmî MCG'nin sıralı ilk 50 skoru, `float64`;
- `ranks`: `1..50`, `int32`.

Bu cache görüntüye bağlıdır; EdgeCAM, Finer-CAM ve ilerideki füzyon deneyleri aynı dosyaları
kullanabilir. Üretim ve değerlendirme komutları Bölüm 12'dedir.

## 3. Git, ortam ve test sırası

İlk kurulum:

```bash
git clone https://github.com/busrakalyon/ClassActivationMap.git
cd ClassActivationMap
uv venv --python 3.12 .venv
source .venv/bin/activate
# PyTorch resmî seçicisindeki uygun CUDA wheel komutunu uv pip ile çalıştırın.
uv pip install -e .
uv pip freeze | tee results/dependencies.freeze.txt
uv run python scripts/audit_environment.py | tee results/environment.json
uv run python -m unittest discover -s tests -v
```

Sonraki çalıştırmalarda güvenli güncelleme ve test:

```bash
uv run python scripts/ubuntu_sync.py --then tests
```

Script yalnız `fetch + ff-only merge` yapar; `reset`, `clean` veya sonuç silme işlemi yoktur.
Tracked yerel değişiklik varsa pull etmeyi reddeder. `results/`, checkpoint, veri, manifest ve
SOP dizinleri Git tarafından izlenmez.

NumPy–PyTorch parite ve ArcFace testlerini özellikle çalıştırmak için:

```bash
uv run python -m unittest tests.test_edgecam_math.EdgeCAMTorchParityTests -v
uv run python -m unittest tests.test_arcface.ArcFaceTests -v
uv run python -m unittest tests.test_deterministic_cam_pipeline.DeterministicCAMPipelineTests -v
```

## 4. Uçtan uca smoke test

Veri gerektirmeden gerçek EfficientNet-B0 + ArcFace üzerinde iki küçük epoch çalıştırır;
checkpoint kaydeder/yükler, EdgeCAM üretir, epoch CAM'lerini birleştirir ve metrik hesaplar:

```bash
uv run python scripts/ubuntu_sync.py --then smoke --device cuda
```

Çıktı `results/smoke/<UTC-zaman>/report.json` altındadır ve Git pull ile silinmez. Sentetik
metrikler bilimsel sonuç değildir. Tam eğitime ancak `status=passed`, iki checkpoint,
`edgecam_finite=true` ve deterministik kontroldeki üç boolean alan `true` olduktan sonra
geçin. EfficientNet kolunda `valid_epochs=0` olması mümkündür; deterministik kontrol bunun
yerine CAM/fusion mekanizmasını zorunlu olarak sınar.

## 5. Manifest, eğitim ve değerlendirme

```bash
uv run python scripts/build_casia_manifest.py \
  --dataset casia_v2 \
  --image-root /data/edgecam/datasets/CASIA2_corrected \
  --mask-root /data/edgecam/datasets/CASIA2_corrected/Gt \
  --output manifests/casia_v2.csv \
  --source-revision "ZIP_SHA256" --validation-fraction 0.10 --seed 42 --hashes

uv run python scripts/audit_manifest.py \
  --manifest manifests/casia_v2.csv \
  --image-root /data/edgecam/datasets/CASIA2_corrected \
  --mask-root /data/edgecam/datasets/CASIA2_corrected/Gt --verify-hashes \
  | tee results/casia_v2_audit.json

uv run python scripts/build_casia_manifest.py \
  --dataset casia_v1 \
  --image-root /data/edgecam/datasets/CASIA1_modified \
  --mask-root /data/edgecam/datasets/CASIA1_groundtruth \
  --authentic-dir Au --tampered-dir Sp \
  --output manifests/casia_v1.csv --source-revision "ZIP_SHA256" --hashes

uv run python scripts/train_edgecam_classifier.py \
  --config configs/experiment1_edgecam.toml \
  --manifest manifests/casia_v2.csv \
  --image-root /data/edgecam/datasets/CASIA2_corrected \
  --mask-root /data/edgecam/datasets/CASIA2_corrected/Gt \
  --output-dir /data/edgecam/checkpoints/experiment1 --device cuda
```

Kullanıcı dizini altındaki düzeltilmiş paket için karşılığı:

```bash
uv run python scripts/train_edgecam_classifier.py \
  --config configs/experiment1_edgecam.toml \
  --manifest manifests/casia_v2.csv \
  --image-root ~/edgecam_data/datasets/CASIA2_corrected/casia \
  --mask-root ~/edgecam_data/datasets/CASIA2_corrected/casia/Gt \
  --output-dir ~/edgecam_data/checkpoints/experiment1 \
  --device cuda
```

`--mask-root` yalnız manifest dosya/hash doğrulamasında kullanılır. Sınıflandırıcı
`ManifestClassificationDataset` üzerinden yalnız görüntü, image-level gerçek/sahte etiketi
ve örnek kimliği alır; maske loss'a veya model girdisine verilmez.

CAM üretimi, validation eşiği ve değerlendirme CLI'larının ayrıntıları `--help` çıktılarında
bulunur. Piksel eşiği CASIAv2 validation'da seçilmeli; CASIAv1'de değiştirilmemelidir.

## 6. Protokol farkları

| Bileşen | Makale | İndirilen kod | Bu altyapı |
|---|---|---|---|
| LR scheduler | `1e-4` periyodik olarak `1e-8`; kilometre taşları yok | Cosine, `T_max=10` | 100-epoch cosine; açık yeniden-uygulama varsayımı |
| Augmentation | Flip, blur, compression | JPEG `p=.5`, flip `p=.5`; blur yorumda; ek CAM-masking yolu var | JPEG/flip + varsayımsal blur `p=.2,k=5`; CAM-masking yok |
| ArcFace inference | Açıklanmamış | Gerçek etiketi margin hesabına verir | Label-free `s*cos(theta)`; sızıntısız fakat koddan farklı |
| CAM sonrası | Denklem 2–4; normalizasyon/final ReLU açık değil | Haritayı kareler, `uint8`, resize | İsteğe bağlı final ReLU + min-max; kare alma yok |
| Epoch füzyonu | Fake olasılığı `>=.5`, max en iyi | Argmax fake, piksel max | Fake olasılığı `>=.5`, float max |
| SOP üretimi | MCG tabanlı, top 50 | Üretici yok; kısmi `seg.py` UCM denemesi | Resmî MCG `accurate`, sıralı top-50; `MCG-reconstructed` olarak raporlanır |
| Validation | Belirtilmemiş | Train/val yolları çoğu config'de aynı | Kaynak-gruplu %10 CASIAv2 val; test tuning yok |

## 7. EdgeCAM matematik diagnostikleri (mevcut Ubuntu ortamı)

Bu bölüm, çalışan `.venv` ortamını değiştirmez ve otomatik bağımlılık senkronizasyonu yapmaz.
Komutlar yalnız CASIAv2 pilot `val` ayrımını okur; checkpoint, manifest, ana EdgeCAM kodu ve
veri seti değiştirilmez. CASIAv1 bu diagnostik seçimlerde kullanılmamalıdır.

Önce kodu ve mevcut ortamı doğrulayın:

```bash
cd ~/Desktop/ClassActivationMap
git pull --ff-only
source .venv/bin/activate
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
python -m unittest discover -s tests -v
```

Özellikle NumPy–PyTorch paritesi, label-free ArcFace ve yeni diagnostik testleri:

```bash
python -m unittest tests.test_edgecam_math.EdgeCAMTorchParityTests -v
python -m unittest tests.test_arcface.ArcFaceTests -v
python -m unittest tests.test_edgecam_diagnostics -v
```

Beklenen toplam test sayısı güncel commit için 62'dir. Windows'taki PyTorch'suz geliştirme
ortamında 53 test geçer ve 9 PyTorch/CUDA testi atlanır; Ubuntu GPU ortamında atlama olmamalıdır.

İlk çalışma için, sabit seed ile seçilen ve iki sınıfı da içeren 8 görüntülük küçük alt küme:

```bash
PILOT_ARCHIVE="$(ls -dt ~/edgecam_archive/preliminary_* | head -n 1)"

python scripts/diagnose_edgecam.py \
  --config "$PILOT_ARCHIVE/configs/experiment1_pilot.toml" \
  --manifest "$PILOT_ARCHIVE/manifests/casia_v2_pilot.csv" \
  --image-root ~/edgecam_data/datasets/CASIA2_corrected/casia \
  --mask-root ~/edgecam_data/datasets/CASIA2_corrected/casia/Gt \
  --checkpoint-dir "$PILOT_ARCHIVE/pilot/training/checkpoints" \
  --output-root "$PILOT_ARCHIVE/diagnostics/edgecam_math" \
  --split val --device cuda --batch-size 4 --limit 8 \
  --batch-consistency-count 8
```

Alt küme tamamlandıktan sonra 128 pilot validation görüntüsünün tamamı:

```bash
python scripts/diagnose_edgecam.py \
  --config "$PILOT_ARCHIVE/configs/experiment1_pilot.toml" \
  --manifest "$PILOT_ARCHIVE/manifests/casia_v2_pilot.csv" \
  --image-root ~/edgecam_data/datasets/CASIA2_corrected/casia \
  --mask-root ~/edgecam_data/datasets/CASIA2_corrected/casia/Gt \
  --checkpoint-dir "$PILOT_ARCHIVE/pilot/training/checkpoints" \
  --output-root "$PILOT_ARCHIVE/diagnostics/edgecam_math" \
  --split val --device cuda --batch-size 4 \
  --batch-consistency-count 8
```

Her çalıştırma `$PILOT_ARCHIVE/diagnostics/edgecam_math/<UTC-zaman>/` altında yeni bir dizin açar; var olan
çıktının üzerine yazmaz. `--keep-epoch-maps` verilmedikçe ara NPZ önbelleği raporlar üretildikten
sonra kaldırılır. Varsayılan sınır bantları 3×3 dilation/erosion ile 1, 2 ve 3 iterasyondur.
Varsayılan eşik grid'i `0.001`–`0.5` aralığındaki genişletilmiş diagnostik listedir. Bunlar yalnız
validation diagnostikleridir ve nihai test sonucu değildir.

Ana raporlar: `summary.json`, `gradient_statistics.csv`, `alpha_statistics.csv`,
`cam_statistics.csv`, `per_image_metrics.csv`, `variant_comparison.csv`,
`threshold_analysis.csv`, `batch_consistency.json`, `gate_comparison.json` ve
`visualizations/`. Ground-truth yalnız bu değerlendirme dosyalarında kullanılır; forward,
gradyan, CAM hesabı veya loss'a girmez.

## 8. Batch-shape sayısal diagnostik ve tam eğitim

Bu test ana EdgeCAM formülünü, `epsilon=1e-8` değerini veya gate davranışını değiştirmez.
Aynı checkpoint ve aynı örnekleri batch 1/4 altında dört CUDA profilinde karşılaştırır:
TF32 açık/kapalı × deterministik açık/kapalı. Stage 2'den embedding ve ArcFace logits'e kadar
katman çıktıları, gradyanlar, ana/diagnostik CAM paritesi, korelasyon, eşik maskesi IoU ve gate
kararları kaydedilir. Aynı görüntünün üç ayrı batch-1 çalıştırması da sınanır.

```bash
cd ~/Desktop/ClassActivationMap
git pull --ff-only
source .venv/bin/activate
PILOT_ARCHIVE="$(ls -dt ~/edgecam_archive/preliminary_* | head -n 1)"

python -m unittest discover -s tests -v

python scripts/diagnose_batch_consistency.py \
  --config "$PILOT_ARCHIVE/configs/experiment1_pilot.toml" \
  --manifest "$PILOT_ARCHIVE/manifests/casia_v2_pilot.csv" \
  --image-root ~/edgecam_data/datasets/CASIA2_corrected/casia \
  --checkpoint "$PILOT_ARCHIVE/pilot/training/checkpoints/epoch_003.pt" \
  --output-root "$PILOT_ARCHIVE/diagnostics/batch_consistency" \
  --split val --samples 4 --repeat-count 3 \
  --thresholds 0.05 0.5
```

CPU float64 referansı daha yavaştır. Gerçek batch 1/4 karşılaştırması için yine dört örnek gerekir:

```bash
python scripts/diagnose_batch_consistency.py \
  --config "$PILOT_ARCHIVE/configs/experiment1_pilot.toml" \
  --manifest "$PILOT_ARCHIVE/manifests/casia_v2_pilot.csv" \
  --image-root ~/edgecam_data/datasets/CASIA2_corrected/casia \
  --checkpoint "$PILOT_ARCHIVE/pilot/training/checkpoints/epoch_003.pt" \
  --output-root "$PILOT_ARCHIVE/diagnostics/batch_consistency" \
  --split val --samples 4 --repeat-count 3 \
  --thresholds 0.05 0.5 --include-cpu-float64
```

Her çalışma yeni UTC dizininde `summary.json` ve `per_sample_comparisons.csv` üretir. Toleranslar
`atol=1e-5`, `rtol=1e-4` olarak raporlanır; test geçsin diye otomatik genişletilmez. Deterministik
algoritmalar sabit işlem şeklinin tekrarlanabilirliğini hedefler, farklı batch şekillerinin bit-eşit
olacağını garanti etmez.

Batch-1 tekrarları kararlı fakat batch 1/4 CAM maskeleri anlamlı biçimde farklıysa bilimsel CAM
çıkarımı batch size 1 ile yapılmalıdır. Bu, eğitim batch size 16 değerini değiştirmez:

```bash
python scripts/generate_edgecam.py \
  --config configs/experiment1_edgecam.toml \
  --manifest manifests/casia_v2.csv \
  --image-root ~/edgecam_data/datasets/CASIA2_corrected/casia \
  --checkpoint results/01_shared_training/checkpoints/epoch_100.pt \
  --output-dir results/02_edgecam/casia_v2_val/materialized_epoch_100 \
  --split val --device cuda --batch-size 1
```

### Sıfırdan 100 epoch eğitim

Komut resume içermez. `build_model(config)` ImageNet başlangıç ağırlıklarını yükler; pilot
checkpoint kullanılmaz. Devam eden eğitime müdahale etmeyin veya bu komutu ikinci kez
başlatmayın. Yalnız gelecekteki temiz bir başlangıç için kayıtlı komut:

```bash
cd ~/Desktop/ClassActivationMap
source .venv/bin/activate
python scripts/train_edgecam_classifier.py \
  --config configs/experiment1_edgecam.toml \
  --manifest manifests/casia_v2.csv \
  --image-root ~/edgecam_data/datasets/CASIA2_corrected/casia \
  --mask-root ~/edgecam_data/datasets/CASIA2_corrected/casia/Gt \
  --output-dir results/01_shared_training --device cuda
```

Her epoch terminalde tek satır JSON olarak yazılır. Aynı değerler `training_metrics.csv` ve
`training_metrics.json` içinde tutulur: epoch, train/validation loss, I-F1, AUC, sensitivity,
specificity, epoch sırasında kullanılan LR ve eğitim+validation süresi. Config'teki
`checkpoint_every=1` nedeniyle `epoch_001.pt`–`epoch_100.pt` korunur; eski dosya otomatik
silinmez. Aynı çıktı dizininde önceden eğitim metriği varsa kod üzerine yazmayı reddeder.

Checkpoint ve CAM alanını gerçek dosyalardan hesaplamak için:

```bash
du -ch results/01_shared_training/checkpoints/epoch_*.pt
python -c "from pathlib import Path; p=list(Path('results/01_shared_training/checkpoints').glob('epoch_*.pt')); a=sum(x.stat().st_size for x in p)/len(p); print('100 checkpoint GiB:', a*100/2**30)"
python -c "print('CASIAv1 100x512x512 float32 GiB:', 1720*100*512*512*4/2**30); print('CASIAv2 val GiB:', 901*100*512*512*4/2**30)"
```

Ham float32 CAM tahmini CASIAv1 için yaklaşık 168 GiB, 901 örneklik CASIAv2 validation için
yaklaşık 88 GiB'dir. Mevcut `np.savez_compressed` kayıpsızdır fakat sıkıştırma oranı haritaya
bağlıdır. Alan sınırlıysa epoch haritalarını kalıcı saklamak yerine checkpoint'leri sırayla okuyup
her örnek için yalnız çalışan piksel maksimumu ve `valid_epochs` sayısını tutan streaming
max-fusion kullanılmalıdır. Bu mod ayrı sonuç dizinine yazılmalı; var olan CAM veya checkpoint
dosyaları silinmemelidir.

## 9. Kayıpsız streaming epoch max-fusion

`stream_edgecam_fusion.py` checkpoint'leri epoch sırasıyla işler fakat tekil epoch CAM'lerini
saklamaz. Her örneğin normalize edilmiş 512×512 float32 çalışan maksimumu, gate'i geçen epoch
sayısı ve en son checkpoint sınıflandırma çıktısı atomik bir durum NPZ'sinde tutulur. Finalde
mevcut araçlarla uyumlu `sample.npy`, `sample__pre_sop.npy`, `sample.json` ve
`image_predictions.csv` üretilir. SOP ve threshold araçlarının girdi sözleşmesi değişmez.

Bilimsel CAM profili iki üretim yolunda da kod tarafından zorunlu uygulanır ve doğrulanır:
batch size 1, float32, AMP kapalı, CUDA matmul TF32 kapalı, cuDNN TF32 kapalı ve model eval.
Gerçek backend değerleri her çalışmanın metadata dosyasına yazılır. Batch size 1 dışında bir
değer verilirse çalışma başlamadan reddedilir.

Checkpoint dizininde yalnız tam olarak `epoch_<integer>.pt` biçimindeki dosyalar kabul edilir.
Epoch'lar config'teki `training.epochs` ile karşılaştırılarak 1'den son epoch'a kadar eksiksiz ve
sayısal sırada doğrulanır. `best_validation_f1.pt` hiçbir zaman ek epoch olarak seçilmez.

### Arşivdeki üç pilot checkpoint ile materyalize/streaming eşdeğerlik testi

İki yol aynı bilimsel backend profilini ve batch size 1'i kullanır. Config, manifest, checkpoint ve
yeni test sonuçları `~/edgecam_archive/preliminary_*` altında kalır; bilimsel `results/` dizinine
diagnostik çıktı yazılmaz:

```bash
cd ~/Desktop/ClassActivationMap
source .venv/bin/activate
PILOT_ARCHIVE="$(ls -dt ~/edgecam_archive/preliminary_* | head -n 1)"
PILOT_CONFIG="$PILOT_ARCHIVE/configs/experiment1_pilot.toml"
PILOT_MANIFEST="$PILOT_ARCHIVE/manifests/casia_v2_pilot.csv"
PILOT_CHECKPOINTS="$PILOT_ARCHIVE/pilot/training/checkpoints"
EQ_ROOT="$PILOT_ARCHIVE/equivalence_tests/streaming_$(date -u +%Y%m%dT%H%M%SZ)"

python scripts/generate_edgecam.py \
  --config "$PILOT_CONFIG" --manifest "$PILOT_MANIFEST" \
  --image-root ~/edgecam_data/datasets/CASIA2_corrected/casia \
  --checkpoint "$PILOT_CHECKPOINTS/epoch_001.pt" \
  --output-dir "$EQ_ROOT/materialized_cams" --split val --device cuda --batch-size 1

python scripts/generate_edgecam.py \
  --config "$PILOT_CONFIG" --manifest "$PILOT_MANIFEST" \
  --image-root ~/edgecam_data/datasets/CASIA2_corrected/casia \
  --checkpoint "$PILOT_CHECKPOINTS/epoch_002.pt" \
  --output-dir "$EQ_ROOT/materialized_cams" --split val --device cuda --batch-size 1

python scripts/generate_edgecam.py \
  --config "$PILOT_CONFIG" --manifest "$PILOT_MANIFEST" \
  --image-root ~/edgecam_data/datasets/CASIA2_corrected/casia \
  --checkpoint "$PILOT_CHECKPOINTS/epoch_003.pt" \
  --output-dir "$EQ_ROOT/materialized_cams" --split val --device cuda --batch-size 1

python scripts/fuse_epoch_cams.py \
  --manifest "$PILOT_MANIFEST" \
  --cam-dir "$EQ_ROOT/materialized_cams/epoch_001" \
  --cam-dir "$EQ_ROOT/materialized_cams/epoch_002" \
  --cam-dir "$EQ_ROOT/materialized_cams/epoch_003" \
  --output-dir "$EQ_ROOT/materialized_fused" --split val --method max --threshold 0.5

python scripts/stream_edgecam_fusion.py \
  --config "$PILOT_CONFIG" --manifest "$PILOT_MANIFEST" \
  --image-root ~/edgecam_data/datasets/CASIA2_corrected/casia \
  --checkpoint-dir "$PILOT_CHECKPOINTS" \
  --output-dir "$EQ_ROOT/streaming_fused" \
  --metadata-output "$EQ_ROOT/metadata/streaming_run.json" \
  --split val --device cuda --batch-size 1 --threshold 0.5

python scripts/compare_fused_outputs.py \
  --manifest "$PILOT_MANIFEST" \
  --reference-dir "$EQ_ROOT/materialized_fused" \
  --candidate-dir "$EQ_ROOT/streaming_fused" \
  --output "$EQ_ROOT/comparison.json" --split val --atol 0 --rtol 0

diff "$EQ_ROOT/materialized_cams/epoch_003/image_predictions.csv" \
  "$EQ_ROOT/streaming_fused/image_predictions.csv"
```

Her materyalize checkpoint dizinindeki `cam_run_metadata.json` ile streaming
`metadata/streaming_run.json` dosyaları aynı profilin gerçekten uygulandığını göstermelidir.
`comparison.json` için beklenen sonuç `status=passed`, maksimum fark `0.0` ve bütün
`valid_epochs` değerlerinin eşit olmasıdır. Fark çıkarsa önce iki yolun batch size ve backend
ayarlarının aynı olduğu doğrulanmalıdır; tolerans yükseltilmemelidir.

### Tam deneyde validation streaming fusion

Batch diagnostikleri nedeniyle bilimsel CAM çıkarımında batch size 1 kullanılır:

```bash
python scripts/stream_edgecam_fusion.py \
  --config configs/experiment1_edgecam.toml \
  --manifest manifests/casia_v2.csv \
  --image-root ~/edgecam_data/datasets/CASIA2_corrected/casia \
  --checkpoint-dir results/01_shared_training/checkpoints \
  --output-dir results/02_edgecam/casia_v2_val \
  --metadata-output results/00_metadata/edgecam_casia_v2_val.json \
  --split val --device cuda --batch-size 1 --threshold 0.5

python scripts/select_pixel_threshold.py \
  --manifest manifests/casia_v2.csv \
  --mask-root ~/edgecam_data/datasets/CASIA2_corrected/casia/Gt \
  --fused-cam-dir results/02_edgecam/casia_v2_val \
  --output results/00_metadata/edgecam_validation_threshold.json --split val \
  --candidates 0.001 0.0025 0.005 0.01 0.015 0.02 0.025 0.03 0.04 0.05 0.075 0.1 0.2 0.3 0.4 0.5
```

CASIAv1 çalıştırmasında `VALIDATION_THRESHOLD`, yalnız yukarıdaki CASIAv2 validation
raporundan alınmalıdır:

```bash
VALIDATION_THRESHOLD="CASIAV2_VALIDATION_DEGERI"

python scripts/stream_edgecam_fusion.py \
  --config configs/experiment1_edgecam.toml \
  --manifest manifests/casia_v1.csv \
  --image-root ~/edgecam_data/datasets/CASIA1_ready \
  --checkpoint-dir results/01_shared_training/checkpoints \
  --output-dir results/02_edgecam/casia_v1_test \
  --metadata-output results/00_metadata/edgecam_casia_v1_test.json \
  --split test --device cuda --batch-size 1 --threshold "$VALIDATION_THRESHOLD"

python scripts/evaluate.py \
  --manifest manifests/casia_v1.csv \
  --mask-root ~/edgecam_data/datasets/CASIA1_groundtruth/CASIA\ 1.0\ groundtruth \
  --prediction-dir results/02_edgecam/casia_v1_test \
  --image-predictions results/02_edgecam/casia_v1_test/image_predictions.csv \
  --output-dir results/02_edgecam/casia_v1_test/evaluation_pre_sop \
  --split test --prediction-suffix __pre_sop.npy --pixel-scope fake
```

MCG-reconstructed SOP akışı ve güncel test komutları Bölüm 12'dedir.

### Güvenli devam

Kesinti sonrasında ilk komutun bütün seçeneklerini aynen koruyup yalnız `--resume` ekleyin:

```bash
python scripts/stream_edgecam_fusion.py \
  --config configs/experiment1_edgecam.toml \
  --manifest manifests/casia_v2.csv \
  --image-root ~/edgecam_data/datasets/CASIA2_corrected/casia \
  --checkpoint-dir results/01_shared_training/checkpoints \
  --output-dir results/02_edgecam/casia_v2_val \
  --metadata-output results/00_metadata/edgecam_casia_v2_val.json \
  --split val --device cuda --batch-size 1 --threshold 0.5 --resume
```

Her örnek güncellemesi geçici dosya + `os.replace` ile atomiktir. Durumdaki
`processed_checkpoint_count` yeniden başlatmada aynı epoch'un iki kez sayılmasını engeller.
Config, manifest, checkpoint hash'leri, batch size, cihaz ve sayısal backend sözleşmesi farklıysa
resume reddedilir. `.streaming_state/` finalden sonra denetim için korunur; kod mevcut durum,
checkpoint veya CAM dosyalarını otomatik silmez.

## 10. CASIAv1/CASIAv2 sızıntı denetimi

Denetim; dosya SHA-256, decode edilmiş RGB SHA-256, 64-bit DCT perceptual hash ve manifestteki
CASIA kaynak kimliklerini ayrı ayrı raporlar. Perceptual-hash eşleşmesi yalnız manuel inceleme
adayıdır ve tek başına test örneğini otomatik dışlamaz. Train–test ve validation–test sayıları ile
gerçek/sahte sınıf çiftleri `summary.json` içinde ayrı tutulur.

Her çalıştırma yeni zaman damgalı bir dizine yazılır; önceki metadata üzerine yazılmaz:

```bash
cd ~/Desktop/ClassActivationMap
source .venv/bin/activate
OVERLAP_OUT="results/00_metadata/dataset_overlap/run_$(date -u +%Y%m%dT%H%M%SZ)"

python scripts/audit_dataset_overlap.py \
  --casia-v2-manifest manifests/casia_v2.csv \
  --casia-v2-image-root ~/edgecam_data/datasets/CASIA2_corrected/casia \
  --casia-v1-manifest manifests/casia_v1.csv \
  --casia-v1-image-root ~/edgecam_data/datasets/CASIA1_ready \
  --output-dir "$OVERLAP_OUT" \
  --phash-max-distance 8
```

Çıktılar:

- `exact_file_sha256.csv`: kodlanmış dosya baytları birebir aynı çiftler.
- `exact_decoded_rgb.csv`: dosya biçimi farklı olsa da boyut ve RGB pikselleri aynı çiftler.
- `perceptual_hash_candidates.csv`: Hamming uzaklığı en fazla 8 olan manuel inceleme adayları.
- `shared_source_ids.csv`: ortak CASIA host/donor kimliği taşıyan türevler.
- `recommended_test_exclusions.csv`: kesin eşleşme veya ortak kaynak nedeniyle önerilen dışlamalar.
- `casia_v1_source_disjoint_proposed.csv`: önerilen dışlama varsa üretilen ayrı test manifesti.

Önerilen manifest ana `manifests/casia_v1.csv` dosyasını değiştirmez. Literatürle karşılaştırılan
özgün CASIAv1 sonucu ve sızıntısız ikincil sonuç ayrı satırlarda raporlanmalıdır. Perceptual-hash
adayları görüntüler yan yana doğrulanmadan temiz manifeste eklenmemelidir.

## 11. Mevcut fused CAM'leri yeniden eşikleme

Bu işlem checkpoint yüklemez ve CAM üretmez. `results/02_edgecam/casia_v2_val/*.npy` içindeki
yerel çözünürlüklü float32 fused CAM'leri okur. Seçim yalnız sahte validation görüntülerindeki
makro P-F1 ile yapılır; eşitlikte önce yüksek IoU, sonra düşük eşik seçilir. Sıfır CAM pozitif
eşiklerde boş kalır.

```bash
python scripts/rethreshold_fused_cams.py \
  --manifest manifests/casia_v2.csv \
  --mask-root ~/edgecam_data/datasets/CASIA2_corrected/casia/Gt \
  --fused-cam-dir results/02_edgecam/casia_v2_val \
  --output-dir results/02_edgecam/casia_v2_selected_threshold \
  --split val \
  --thresholds 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.075 0.1 0.15 0.5

python scripts/evaluate.py \
  --manifest manifests/casia_v2.csv \
  --mask-root ~/edgecam_data/datasets/CASIA2_corrected/casia/Gt \
  --prediction-dir results/02_edgecam/casia_v2_selected_threshold/masks \
  --image-predictions results/02_edgecam/casia_v2_val/image_predictions.csv \
  --output-dir results/02_edgecam/casia_v2_selected_threshold/evaluation \
  --split val --prediction-suffix __pre_sop.npy --pixel-scope fake
```

`threshold_analysis.csv` her eşik için P-F1, IoU, sahte örneklerde ortalama tahmini alan,
tüm validation kümesindeki tahmini alan ve gerçek manipülasyon alanı oranını içerir.
`selection.json` tamamen boş/tam dolu referanslarını ve seçilen eşiği kaydeder. Yalnız seçilen
eşik maskeleri `masks/` altına yazılır. Çıktı dizini doluysa betik üzerine yazmayı reddeder.

CASIAv1 testinde yeni eşik seçimi yapılmaz. Validation'da seçilen değer CASIAv1 komutuna sabit
olarak aktarılır.

## 12. MCG-reconstructed SOP ile EdgeCAM baseline

### 12.1 Ortam değişkenleri

```bash
cd ~/Desktop/ClassActivationMap
source .venv/bin/activate

MCG_ROOT=~/edgecam_tools/mcg/pre-trained
MATLAB_BIN=~/edgecam_tools/MATLAB/bin/matlab
SOP_ROOT=results/06_shared_sop/mcg_reconstructed

test -x "$MATLAB_BIN"
test -f "$MCG_ROOT/im2mcg.m"
git -C "$MCG_ROOT" rev-parse HEAD
```

MATLAB her parçada yalnız sıralı ilk 50 binary maskeyi üretir. Python bunları
`np.savez_compressed` ile kayıpsız NPZ cache'e dönüştürür ve başarılı dönüşümden sonra büyük
geçici binary dosyaları kaldırır. Varsayılan parça büyüklüğü 10 görüntüdür. Kesinti sonrasında
aynı komut tekrar çalıştırılır; geçerli NPZ+metadata çiftleri hash ve şekil kontrolünden sonra
atlanır, tamamlanmış MATLAB ara çıktıları finalize edilir.

Resmî `im2mcg.m`, 2.5 megapiksel üzerindeki görüntülerde istenen `accurate` modu sessizce
`fast` moda çevirir. Cache üreticisi yanlış metadata oluşmaması için bu durumda örneği hata
olarak kaydeder; otomatik resize veya SCG fallback uygulamaz.

### 12.2 CASIAv2 validation proposal cache

```bash
python scripts/generate_mcg_proposals.py \
  --manifest manifests/casia_v2.csv \
  --image-root ~/edgecam_data/datasets/CASIA2_corrected/casia \
  --split val \
  --output-dir "$SOP_ROOT/casia_v2_val" \
  --metadata-dir "$SOP_ROOT/metadata/casia_v2_val" \
  --mcg-root "$MCG_ROOT" \
  --matlab-bin "$MATLAB_BIN" \
  --mode accurate --top-k 50 --chunk-size 10

python scripts/generate_mcg_proposals.py \
  --manifest manifests/casia_v2.csv \
  --image-root ~/edgecam_data/datasets/CASIA2_corrected/casia \
  --split val \
  --output-dir "$SOP_ROOT/casia_v2_val" \
  --metadata-dir "$SOP_ROOT/metadata/casia_v2_val" \
  --mcg-root "$MCG_ROOT" \
  --matlab-bin "$MATLAB_BIN" \
  --mode accurate --top-k 50 --verify-only
```

İkinci komut 901 örneğin tamamında dosya, görüntü SHA-256, çözünürlük, top-50 sayısı, binary
değerler, ranking ve metadata sözleşmesini yeniden doğrular. Örnek metadata; toplam MCG
proposal sayısı, MATLAB sürümü, resmî MCG kaynak yolu/commit'i ve NPZ yolunu içerir.

### 12.3 EdgeCAM SOP scoring

Ground-truth ve görüntü etiketi bu komuta verilmez. Seçim, sürekli float32 fused CAM ile makale
Denklem 7–8'deki `sum(proposal_boundary * fused_cam)` skoruna göre yapılır:

```bash
python scripts/apply_sop.py \
  --manifest manifests/casia_v2.csv \
  --fused-cam-dir results/02_edgecam/casia_v2_val \
  --proposal-dir "$SOP_ROOT/casia_v2_val" \
  --proposal-metadata-dir "$SOP_ROOT/metadata/casia_v2_val" \
  --proposal-format mcg_hwk \
  --output-dir results/02_edgecam/casia_v2_val_sop \
  --split val --expected-count 50 --kernel-size 3 --iterations 1
```

Kesinti olursa aynı seçeneklerle `--resume` eklenir. Mevcut fused CAM dosyaları değiştirilmez.
`kernel-size=3` ve tek morfoloji iterasyonu yeniden-uygulama parametreleridir; makale Denklem 7
genişletme/erozyon işlemlerini tanımlar fakat structuring-element ayrıntısını belirtmez.

### 12.4 Validation değerlendirme ve görseller

```bash
python scripts/evaluate.py \
  --manifest manifests/casia_v2.csv \
  --mask-root ~/edgecam_data/datasets/CASIA2_corrected/casia/Gt \
  --prediction-dir results/02_edgecam/casia_v2_val_sop \
  --image-predictions results/02_edgecam/casia_v2_val/image_predictions.csv \
  --output-dir results/02_edgecam/casia_v2_val_sop_evaluation \
  --split val --prediction-suffix __post_sop.npy --pixel-scope fake

python scripts/visualize_sop_results.py \
  --manifest manifests/casia_v2.csv \
  --image-root ~/edgecam_data/datasets/CASIA2_corrected/casia \
  --mask-root ~/edgecam_data/datasets/CASIA2_corrected/casia/Gt \
  --fused-cam-dir results/02_edgecam/casia_v2_val \
  --prediction-dir results/02_edgecam/casia_v2_val_sop \
  --output-dir results/02_edgecam/casia_v2_val_sop_visualizations \
  --split val --sample-count 20
```

`evaluate.py` P-F1, IoU, I-F1, AUC ve C-F1 üretir. Ground-truth yalnız bu değerlendirme ve
görselleştirme adımlarında kullanılır. Görselleştirme renk dönüşümü metrik girdilerini değiştirmez.

### 12.5 CASIAv1 test karar ve çalışma sırası

Önce Bölüm 10'daki overlap denetimi çalıştırılır. Karar kuralı:

- Literatürdeki EdgeCAM protokolüyle doğrudan karşılaştırılacak ana satır
  `manifests/casia_v1.csv` ile üretilir.
- `recommended_test_exclusions.csv` boş değilse
  `casia_v1_source_disjoint_proposed.csv` değiştirilmeden korunur ve ayrıca sızıntı-kontrollü
  ikincil test olarak raporlanır.
- Perceptual-hash-only adayları manuel doğrulama olmadan dışlanmaz.

CASIAv1'in tamamı için MCG cache yalnız bir kez üretilir; temiz alt manifest gerekirse aynı cache'i
yeniden kullanır:

```bash
python scripts/generate_mcg_proposals.py \
  --manifest manifests/casia_v1.csv \
  --image-root ~/edgecam_data/datasets/CASIA1_ready \
  --split test \
  --output-dir "$SOP_ROOT/casia_v1_test" \
  --metadata-dir "$SOP_ROOT/metadata/casia_v1_test" \
  --mcg-root "$MCG_ROOT" \
  --matlab-bin "$MATLAB_BIN" \
  --mode accurate --top-k 50 --chunk-size 10
```

Ana literatür karşılaştırması için:

```bash
TEST_MANIFEST=manifests/casia_v1.csv
TEST_TAG=casia_v1_test
VALIDATION_THRESHOLD=$(python -c "import json; print(json.load(open('results/02_edgecam/casia_v2_selected_threshold/selection.json'))['threshold'])")

python scripts/stream_edgecam_fusion.py \
  --config configs/experiment1_edgecam.toml \
  --manifest "$TEST_MANIFEST" \
  --image-root ~/edgecam_data/datasets/CASIA1_ready \
  --checkpoint-dir results/01_shared_training/checkpoints \
  --output-dir "results/02_edgecam/$TEST_TAG" \
  --metadata-output "results/00_metadata/edgecam_${TEST_TAG}.json" \
  --split test --device cuda --batch-size 1 --threshold "$VALIDATION_THRESHOLD"

python scripts/apply_sop.py \
  --manifest "$TEST_MANIFEST" \
  --fused-cam-dir "results/02_edgecam/$TEST_TAG" \
  --proposal-dir "$SOP_ROOT/casia_v1_test" \
  --proposal-metadata-dir "$SOP_ROOT/metadata/casia_v1_test" \
  --proposal-format mcg_hwk \
  --output-dir "results/02_edgecam/${TEST_TAG}_sop" \
  --split test --expected-count 50 --kernel-size 3 --iterations 1

python scripts/evaluate.py \
  --manifest "$TEST_MANIFEST" \
  --mask-root ~/edgecam_data/datasets/CASIA1_groundtruth/CASIA\ 1.0\ groundtruth \
  --prediction-dir "results/02_edgecam/${TEST_TAG}_sop" \
  --image-predictions "results/02_edgecam/$TEST_TAG/image_predictions.csv" \
  --output-dir "results/02_edgecam/${TEST_TAG}_sop_evaluation" \
  --split test --prediction-suffix __post_sop.npy --pixel-scope fake
```

Source-disjoint ikincil test çalıştırılacaksa yalnız `TEST_MANIFEST` ve `TEST_TAG` yeni bir shell'de
önerilen manifestin gerçek yoluna göre değiştirilir; original ve temiz test çıktıları aynı dizine
yazılmaz. CASIAv1 üzerinde eşik veya başka hiperparametre seçilmez.
