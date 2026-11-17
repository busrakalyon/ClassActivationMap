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

Makalenin kaynağı MCG'dir: Pont-Tuset vd., *Multiscale Combinatorial Grouping*, TPAMI
39(1), 2017. Makale top-50 segment önerisini kullanır fakat MCG model varyantı, sıralama ve
ön/son işleme ayrıntılarını yayımlamaz.

Öncelik sırası:

1. Yazarın resmî Baidu paketi: `https://pan.baidu.com/s/1PXUHecO7TT66DUikOh4epg?pwd=vfms`
   (çıkarma kodu `vfms`). İndirildikten sonra arşiv hash'i ve dosya biçimi kaydedilmeli.
2. Erişim yoksa yazarlara `yzhoulv@foxmail.com` üzerinden CASIAv1/CASIAv2 SOP artefaktı,
   MCG sürümü ve top-50 üretim komutu sorulmalı.
3. Resmî MCG kodu `https://github.com/jponttuset/mcg` ile yeniden üretim yapılabilir; fakat
   bu sonuç “özgün SOP” değil, “MCG-reconstruction” olarak ayrı raporlanmalıdır.

Resmî MCG paketinin önceden eğitilmiş sürümü MATLAB ve bazı derlenmiş bileşenlere dayanır;
yalnız `uv` ile kurulabilen saf Python çözümü değildir. Bu Ubuntu hesabında MATLAB/runtime
yoksa SOP'yi yeniden üretmenin güvenilir yolu yazar artefaktını edinmektir. Yerel özgün
`seg.py` içindeki UCM eşikleme/connected-components denemesi de makaledeki top-50 MCG
üreticisinin uygulaması değildir.

Mevcut `apply_sop.py`, yalnız doğrulanmış `[50,H,W]` NPZ önerilerini skorlar. Baidu
paketinin biçimi görülmeden otomatik dönüştürücü yazılmamıştır.

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
```

## 4. Uçtan uca smoke test

Veri gerektirmeden gerçek EfficientNet-B0 + ArcFace üzerinde iki küçük epoch çalıştırır;
checkpoint kaydeder/yükler, EdgeCAM üretir, epoch CAM'lerini birleştirir ve metrik hesaplar:

```bash
uv run python scripts/ubuntu_sync.py --then smoke --device cuda
```

Çıktı `results/smoke/<UTC-zaman>/report.json` altındadır ve Git pull ile silinmez. Sentetik
metrikler bilimsel sonuç değildir. Tam eğitime ancak `status=passed`, iki checkpoint ve
`edgecam_finite=true` görüldükten sonra geçin.

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
  --output-dir /data/edgecam/checkpoints/experiment1 --device cuda
```

CAM üretimi, validation eşiği, CASIAv1 testi ve SOP komutları ana `README.md` içindedir.
Piksel eşiği CASIAv2 validation'da seçilmeli; CASIAv1'de değiştirilmemelidir.

## 6. Protokol farkları

| Bileşen | Makale | İndirilen kod | Bu altyapı |
|---|---|---|---|
| LR scheduler | `1e-4` periyodik olarak `1e-8`; kilometre taşları yok | Cosine, `T_max=10` | 100-epoch cosine; açık yeniden-uygulama varsayımı |
| Augmentation | Flip, blur, compression | JPEG `p=.5`, flip `p=.5`; blur yorumda; ek CAM-masking yolu var | JPEG/flip + varsayımsal blur `p=.2,k=5`; CAM-masking yok |
| ArcFace inference | Açıklanmamış | Gerçek etiketi margin hesabına verir | Label-free `s*cos(theta)`; sızıntısız fakat koddan farklı |
| CAM sonrası | Denklem 2–4; normalizasyon/final ReLU açık değil | Haritayı kareler, `uint8`, resize | İsteğe bağlı final ReLU + min-max; kare alma yok |
| Epoch füzyonu | Fake olasılığı `>=.5`, max en iyi | Argmax fake, piksel max | Fake olasılığı `>=.5`, float max |
| SOP üretimi | MCG tabanlı, top 50 | Üretici yok; kısmi `seg.py` UCM denemesi | Üretici yok; yalnız doğrulanmış 50 öneriyi skorlar |
| Validation | Belirtilmemiş | Train/val yolları çoğu config'de aynı | Kaynak-gruplu %10 CASIAv2 val; test tuning yok |
