# Muc 8 - Ban dien nhap tu du lieu thuc (Jena)

- Che do tinh Delta/nguong: gia tri tho
- Tan suat: 1 gio (lay 1 trong moi 6 dong). So dong: 70092. Khoang trong >1h: 5, >6h: 2
- Loi cam bien gio (< -900) da gan ve 0: 7 dong

## 1. Cach xu ly chu ky ngay/mua

| channel | acf_lag24 | r2_hour | r2_month | r2_month_hour |
|---|---|---|---|---|
| T (degC) | 0.9214 | 0.0678 | 0.6307 | 0.7105 |

- L = 96 = 4.0 x 24 gio (L la boi cua chu ky ngay: True).
- Quyet dinh: ___ (A: phan du khi hau / B: L boi chu ky ngay)

## 2. L va H

- L = 96, H = 24. So cua so: 69973 (train 48981, val 10496, test 10496).

## 3. Kenh tham gia luat

- Co trong dataset: pressure=co, temp=co, humidity=co, wind=co

## 4. Kenh muc tieu: T (degC)

## 5. Nguong cu the (chi tinh tren train)

| pattern | channel | feature | op | quantile | threshold |
|---|---|---|---|---|---|
| pressure_drop | p (mbar) | delta | <= | 0.1000 | -1.4000 |
| rapid_cooling | T (degC) | delta | <= | 0.1000 | -2.9200 |
| rapid_warming | T (degC) | delta | >= | 0.9000 | 3.1500 |
| heat_wave | T (degC) | roll_mean | >= | 0.9000 | 19.2308 |
| humidity_surge | rh (%) | delta | >= | 0.9000 | 12.3000 |
| wind_gust | wv (m/s) | level | >= | 0.9900 | 7.1900 |

## 6. Tan suat nhan ung vien (so nhan cuoi: 4-7, chot sau khi co ket qua C2, C3)

| pattern | train_share | train_n | val_share | val_n | test_share | test_n |
|---|---|---|---|---|---|---|
| pressure_drop | 0.1003 | 4911 | 0.1012 | 1062 | 0.0912 | 957 |
| rapid_cooling | 0.1004 | 4919 | 0.0984 | 1033 | 0.0872 | 915 |
| rapid_warming | 0.1001 | 4904 | 0.1032 | 1083 | 0.0904 | 949 |
| heat_wave | 0.1002 | 4906 | 0.1199 | 1258 | 0.1207 | 1267 |
| humidity_surge | 0.1005 | 4922 | 0.1008 | 1058 | 0.0897 | 941 |
| wind_gust | 0.0100 | 492 | 0.0129 | 135 | 0.0146 | 153 |
| stable | 0.6345 | 31076 | 0.6208 | 6516 | 0.6482 | 6803 |

## 7. Chong lan: P(cot | hang) tren train

| given | pressure_drop | rapid_cooling | rapid_warming | heat_wave | humidity_surge | wind_gust |
|---|---|---|---|---|---|---|
| pressure_drop | 1.0000 | 0.0210 | 0.1865 | 0.0784 | 0.0446 | 0.0411 |
| rapid_cooling | 0.0209 | 1.0000 | 0.0000 | 0.2131 | 0.7099 | 0.0018 |
| rapid_warming | 0.1868 | 0.0000 | 1.0000 | 0.1578 | 0.0000 | 0.0108 |
| heat_wave | 0.0785 | 0.2136 | 0.1578 | 1.0000 | 0.1792 | 0.0035 |
| humidity_surge | 0.0445 | 0.7095 | 0.0000 | 0.1786 | 1.0000 | 0.0030 |
| wind_gust | 0.4106 | 0.0183 | 0.1077 | 0.0346 | 0.0305 | 1.0000 |

- Thu tu uu tien khi chong lan: ___
