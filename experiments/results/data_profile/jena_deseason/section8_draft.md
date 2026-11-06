# Muc 8 - Ban dien nhap tu du lieu thuc (Jena)

- Che do tinh Delta/nguong: phan du khi hau (deseason theo thang x gio)
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
| pressure_drop | p (mbar) | delta | <= | 0.1000 | -1.2464 |
| rapid_cooling | T (degC) | delta | <= | 0.1000 | -1.8829 |
| rapid_warming | T (degC) | delta | >= | 0.9000 | 1.8578 |
| heat_wave | T (degC) | roll_mean | >= | 0.9000 | 5.4473 |
| humidity_surge | rh (%) | delta | >= | 0.9000 | 8.4756 |
| wind_gust | wv (m/s) | level | >= | 0.9900 | 4.6505 |

## 6. Tan suat nhan ung vien (so nhan cuoi: 4-7, chot sau khi co ket qua C2, C3)

| pattern | train_share | train_n | val_share | val_n | test_share | test_n |
|---|---|---|---|---|---|---|
| pressure_drop | 0.0998 | 4889 | 0.1019 | 1070 | 0.0912 | 957 |
| rapid_cooling | 0.1001 | 4904 | 0.0986 | 1035 | 0.0978 | 1026 |
| rapid_warming | 0.1001 | 4903 | 0.1004 | 1054 | 0.0864 | 907 |
| heat_wave | 0.1002 | 4906 | 0.1176 | 1234 | 0.1612 | 1692 |
| humidity_surge | 0.1001 | 4904 | 0.1023 | 1074 | 0.0917 | 962 |
| wind_gust | 0.0100 | 491 | 0.0128 | 134 | 0.0155 | 163 |
| stable | 0.6218 | 30455 | 0.6050 | 6350 | 0.5958 | 6254 |

## 7. Chong lan: P(cot | hang) tren train

| given | pressure_drop | rapid_cooling | rapid_warming | heat_wave | humidity_surge | wind_gust |
|---|---|---|---|---|---|---|
| pressure_drop | 1.0000 | 0.0454 | 0.1937 | 0.1313 | 0.0785 | 0.0458 |
| rapid_cooling | 0.0453 | 1.0000 | 0.0000 | 0.1458 | 0.5863 | 0.0098 |
| rapid_warming | 0.1931 | 0.0000 | 1.0000 | 0.0775 | 0.0049 | 0.0133 |
| heat_wave | 0.1309 | 0.1457 | 0.0775 | 1.0000 | 0.1292 | 0.0338 |
| humidity_surge | 0.0783 | 0.5863 | 0.0049 | 0.1293 | 1.0000 | 0.0128 |
| wind_gust | 0.4562 | 0.0978 | 0.1324 | 0.3381 | 0.1283 | 1.0000 |

- Thu tu uu tien khi chong lan: ___
