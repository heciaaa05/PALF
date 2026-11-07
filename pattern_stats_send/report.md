# Pattern stats - Jena

Chi do, khong chon thay nguoi dung. Khong co ket luan nen dung cach nao.
Khi hau, phan vi, diem z va ACF chi tinh tren train sau purge gap. Val/test chi ap dung nguong do.
Dac trung va nhan chi dung X. Don vi vat ly, sau khi gan o `wv` / `max. wv` < -900 ve 0.

- So dong gio: 70092; so o wv/max.wv < -900 da gan 0: 7
- L = 96, H = 24, purge gap = 119 cua so moi ranh gioi
- Cua so: train 48981, val 10377, test 10377 (tong 69973)
- Khi hau: 49076 dong dau, khoa month_hour
- A_k = mean(x - khi hau) trong cua so. Cach B, heat/cold dung mean tho cua T.
- Kenh spike (tham so, khong phai ket luan): T. Phan vi: spike 0.99, thap 0.1, cao 0.9.
- Seed mau cua so: 0. Khong co buoc ngau nhien nao khac.
- C2 dat khi share_train >= 2% va min(n) qua 3 tap >= 300. Cot 5% chi la moc trong, khong doi ket qua dat/khong dat.
- C3 canh bao khi Normal > 70% o bat ky tap. Cot 60% la moc trong.
- C4: |share - share_train| / share_train <= 0.5 (tham so cua script; prompt khong chot so nay).

## 1. Chu ky ngay/mua

r2 = ty le phuong sai chuoi gio giai thich boi trung binh nhom. acf = tu tuong quan.
corr_delta_raw_end_hour = Pearson giua Delta tho va gio ket thuc (gio la so 0-23).
eta2_delta = ty le phuong sai cua Delta giai thich boi gio ket thuc, gio duoc xem la nhom.

| channel | r2_hour | r2_month_hour | acf_lag24 | acf_lag96 | corr_delta_raw_end_hour | eta2_delta_raw | eta2_delta_resid |
|---|---|---|---|---|---|---|---|
| p (mbar) | 0.0015 | 0.0263 | 0.7412 | 0.2737 | -0.0032 | 0.0002 | 0.0000 |
| T (degC) | 0.0678 | 0.7105 | 0.9214 | 0.8091 | 0.0490 | 0.0130 | 0.0000 |
| rh (%) | 0.2608 | 0.4570 | 0.6647 | 0.5084 | -0.0618 | 0.0191 | 0.0000 |
| wv (m/s) | 0.1178 | 0.1532 | 0.3179 | 0.1694 | 0.0384 | 0.0061 | 0.0000 |

Delta tho va Delta phan du:

| channel | std_raw | std_resid | std_ratio | std_raw_minus_resid | clim_std_ratio | corr_raw_resid | iqr_raw | iqr_resid | p01_raw | p99_raw | p01_resid | p99_resid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p (mbar) | 9.9826 | 9.9632 | 0.9981 | 0.6414 | 0.0643 | 0.9979 | 12.4200 | 12.4218 | -25.4620 | 23.7020 | -25.3159 | 23.6082 |
| T (degC) | 5.3785 | 5.4910 | 1.0209 | 1.4975 | 0.2784 | 0.9622 | 7.1700 | 7.3162 | -12.2920 | 13.2800 | -12.6532 | 13.4392 |
| rh (%) | 16.6198 | 16.4231 | 0.9882 | 3.1425 | 0.1891 | 0.9820 | 20.0000 | 19.7334 | -41.6840 | 43.0920 | -41.4220 | 42.8723 |
| wv (m/s) | 1.9840 | 1.9721 | 0.9940 | 0.2061 | 0.1039 | 0.9946 | 2.1800 | 2.1566 | -5.2500 | 5.3200 | -5.1864 | 5.2963 |

Cua so co 96 diem nen moc ket thuc lech moc bat dau 95 gio; (95) mod 24 = 23. Gio ket thuc = gio bat dau + 23 (mod 24).
eta2 cua Delta tho theo gio ket thuc nam trong [0.0002, 0.0191].
std(Delta phan du) / std(Delta tho) nam trong [0.9882, 1.0209].
std(Delta tho - Delta phan du) / std(Delta tho) nam trong [0.0643, 0.2784]. Hieu nay bang khi hau tai diem cuoi tru khi hau tai diem dau.

## 2. Phan vi dac trung tren train

A (anom) la mean(x - khi hau) trong cua so, chi co o cot mode = resid.

| channel | feature | mode | p01 | p05 | p10 | p25 | p50 | p75 | p90 | p95 | p99 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| p (mbar) | delta | raw | -25.4620 | -16.1500 | -12.1900 | -6.1500 | 0.1000 | 6.2700 | 12.1600 | 16.3100 | 23.7020 |
| p (mbar) | delta | resid | -25.3159 | -16.2784 | -12.2190 | -6.1479 | 0.0634 | 6.2739 | 12.1577 | 16.2794 | 23.6082 |
| p (mbar) | jump | raw | 0.5600 | 0.6600 | 0.7200 | 0.8600 | 1.0900 | 1.4200 | 1.9400 | 2.3600 | 3.6900 |
| p (mbar) | jump | resid | 0.4516 | 0.5623 | 0.6390 | 0.7984 | 1.0607 | 1.4540 | 2.1759 | 2.7871 | 3.7137 |
| p (mbar) | sigma | raw | 0.9620 | 1.4973 | 1.8720 | 2.6058 | 3.7036 | 5.1757 | 6.9560 | 8.1687 | 10.8956 |
| p (mbar) | sigma | resid | 0.8712 | 1.4475 | 1.8044 | 2.5900 | 3.7034 | 5.1841 | 6.9442 | 8.1335 | 10.8095 |
| p (mbar) | anom | resid | -17.0772 | -11.9000 | -8.5805 | -3.8866 | 0.0157 | 4.1970 | 8.2682 | 11.3433 | 16.1136 |
| T (degC) | delta | raw | -12.2920 | -8.8600 | -6.8100 | -3.5500 | 0.0700 | 3.6200 | 6.8500 | 8.9100 | 13.2800 |
| T (degC) | delta | resid | -12.6532 | -9.0505 | -7.0098 | -3.6562 | 0.0123 | 3.6600 | 6.9534 | 9.0612 | 13.4392 |
| T (degC) | jump | raw | 0.7900 | 1.4000 | 1.8000 | 2.5000 | 3.4200 | 4.1200 | 5.0600 | 5.8200 | 8.2400 |
| T (degC) | jump | resid | 0.8845 | 1.3015 | 1.5585 | 1.9949 | 2.7617 | 3.5800 | 4.8036 | 5.6779 | 7.6911 |
| T (degC) | sigma | raw | 1.0010 | 1.5950 | 1.9632 | 2.6016 | 3.4531 | 4.3080 | 5.0250 | 5.4466 | 6.2302 |
| T (degC) | sigma | resid | 1.0905 | 1.4293 | 1.6296 | 1.9891 | 2.5133 | 3.1341 | 3.8430 | 4.3404 | 5.4327 |
| T (degC) | anom | resid | -10.3126 | -6.2179 | -4.6102 | -2.2134 | 0.0328 | 2.5691 | 4.7405 | 5.8480 | 7.6897 |
| rh (%) | delta | raw | -41.6840 | -27.5900 | -20.5000 | -10.0000 | -0.2200 | 10.0000 | 20.6900 | 27.5700 | 43.0920 |
| rh (%) | delta | resid | -41.4220 | -27.0055 | -20.2445 | -9.8571 | -0.2434 | 9.8763 | 20.4280 | 27.3010 | 42.8723 |
| rh (%) | jump | raw | 5.2000 | 7.8000 | 9.4000 | 12.2300 | 16.1900 | 20.7200 | 26.9000 | 30.4100 | 38.9700 |
| rh (%) | jump | resid | 4.4786 | 6.7055 | 7.9028 | 10.4820 | 14.4678 | 19.2554 | 25.1923 | 28.6007 | 35.6820 |
| rh (%) | sigma | raw | 3.3913 | 5.2844 | 6.5604 | 9.3869 | 13.4827 | 16.2167 | 18.2120 | 19.2585 | 21.1698 |
| rh (%) | sigma | resid | 3.3304 | 4.8920 | 5.6777 | 7.2257 | 9.2290 | 11.3431 | 13.4302 | 14.8606 | 17.5212 |
| rh (%) | anom | resid | -17.3548 | -11.8684 | -8.9306 | -4.8103 | 0.1883 | 4.8075 | 8.8684 | 11.5553 | 16.4075 |
| wv (m/s) | delta | raw | -5.2500 | -3.2300 | -2.3500 | -1.0900 | 0.0000 | 1.0900 | 2.3400 | 3.2300 | 5.3200 |
| wv (m/s) | delta | resid | -5.1864 | -3.2179 | -2.3472 | -1.0841 | 0.0043 | 1.0725 | 2.3240 | 3.2153 | 5.2963 |
| wv (m/s) | jump | raw | 1.5300 | 1.8700 | 2.0700 | 2.4700 | 3.1000 | 3.9000 | 4.8700 | 5.5800 | 6.4900 |
| wv (m/s) | jump | resid | 1.4948 | 1.8167 | 2.0454 | 2.4255 | 3.0019 | 3.8387 | 4.7854 | 5.5081 | 6.3788 |
| wv (m/s) | sigma | raw | 0.6203 | 0.7555 | 0.8400 | 1.0154 | 1.2584 | 1.5600 | 1.8505 | 2.1050 | 2.5465 |
| wv (m/s) | sigma | resid | 0.5889 | 0.6975 | 0.7656 | 0.9057 | 1.1151 | 1.3919 | 1.7142 | 1.9991 | 2.4444 |
| wv (m/s) | anom | resid | -1.2324 | -0.8897 | -0.7044 | -0.4118 | -0.0780 | 0.3350 | 0.8138 | 1.1310 | 1.9434 |

Histogram (train, truc tan suat log): `figures/feat_hist_<p|T|rh|wv>.png`.

## Cach A - dac trung phan du so voi khi hau

Thu tu uu tien: spike > pressure_drop > humidity_rise > heat_anomaly > cold_anomaly > normal

### 3. Nguong va ty le kich hoat (truoc uu tien)

| rule | feature | op | quantile | threshold |
|---|---|---|---|---|
| spike | T__jump__resid | >= | 0.9900 | 7.6911 |
| pressure_drop | p__delta__resid | <= | 0.1000 | -12.2190 |
| humidity_rise | rh__delta__resid | >= | 0.9000 | 20.4280 |
| heat_anomaly | T__anom | >= | 0.9000 | 4.7405 |
| cold_anomaly | T__anom | <= | 0.1000 | -4.6102 |

| rule | share_train | share_val | share_test |
|---|---|---|---|
| spike | 0.0116 | 0.0092 | 0.0000 |
| pressure_drop | 0.1000 | 0.1032 | 0.0895 |
| humidity_rise | 0.1000 | 0.0916 | 0.0799 |
| heat_anomaly | 0.1000 | 0.1245 | 0.1813 |
| cold_anomaly | 0.1000 | 0.0340 | 0.0241 |

### 4. Ma tran chong lan tren train (so cua so; duong cheo = so cua so cua luat)

| rule | spike | pressure_drop | humidity_rise | heat_anomaly | cold_anomaly |
|---|---|---|---|---|---|
| spike | 570 | 29 | 42 | 94 | 3 |
| pressure_drop | 29 | 4899 | 521 | 600 | 591 |
| humidity_rise | 42 | 521 | 4899 | 642 | 301 |
| heat_anomaly | 94 | 600 | 642 | 4899 | 0 |
| cold_anomaly | 3 | 591 | 301 | 0 | 4899 |

P(cot | hang) tren train:

| rule | spike | pressure_drop | humidity_rise | heat_anomaly | cold_anomaly |
|---|---|---|---|---|---|
| spike | 1.000 | 0.051 | 0.074 | 0.165 | 0.005 |
| pressure_drop | 0.006 | 1.000 | 0.106 | 0.122 | 0.121 |
| humidity_rise | 0.009 | 0.106 | 1.000 | 0.131 | 0.061 |
| heat_anomaly | 0.019 | 0.122 | 0.131 | 1.000 | 0.000 |
| cold_anomaly | 0.001 | 0.121 | 0.061 | 0.000 | 1.000 |

### 5. Nhan sau uu tien, va tieu chi C2-C4

| label | n_train | share_train | n_val | share_val | n_test | share_test |
|---|---|---|---|---|---|---|
| spike | 570 | 0.0116 | 95 | 0.0092 | 0 | 0.0000 |
| pressure_drop | 4870 | 0.0994 | 1071 | 0.1032 | 929 | 0.0895 |
| humidity_rise | 4336 | 0.0885 | 862 | 0.0831 | 776 | 0.0748 |
| heat_anomaly | 3702 | 0.0756 | 1058 | 0.1020 | 1511 | 0.1456 |
| cold_anomaly | 4049 | 0.0827 | 279 | 0.0269 | 219 | 0.0211 |
| normal | 31454 | 0.6422 | 7012 | 0.6757 | 6942 | 0.6690 |

| label | share_train | n_train | n_val | n_test | min_n | share>=2% | share>=5% | min_n>=300 | C2 |
|---|---|---|---|---|---|---|---|---|---|
| spike | 0.0116 | 570 | 95 | 0 | 0 | False | False | False | False |
| pressure_drop | 0.0994 | 4870 | 1071 | 929 | 929 | True | True | True | True |
| humidity_rise | 0.0885 | 4336 | 862 | 776 | 776 | True | True | True | True |
| heat_anomaly | 0.0756 | 3702 | 1058 | 1511 | 1058 | True | True | True | True |
| cold_anomaly | 0.0827 | 4049 | 279 | 219 | 219 | True | True | False | False |

| normal_train | normal_val | normal_test | all<=70% | all<=60% |
|---|---|---|---|---|
| 0.6422 | 0.6757 | 0.6690 | True | False |

| label | share_train | share_val | share_test | rel_dev_val | rel_dev_test | C4 |
|---|---|---|---|---|---|---|
| spike | 0.0116 | 0.0092 | 0.0000 | 0.2133 | 1.0000 | False |
| pressure_drop | 0.0994 | 0.1032 | 0.0895 | 0.0380 | 0.0996 | True |
| humidity_rise | 0.0885 | 0.0831 | 0.0748 | 0.0616 | 0.1552 | True |
| heat_anomaly | 0.0756 | 0.1020 | 0.1456 | 0.3490 | 0.9266 | False |
| cold_anomaly | 0.0827 | 0.0269 | 0.0211 | 0.6748 | 0.7447 | False |
| normal | 0.6422 | 0.6757 | 0.6690 | 0.0523 | 0.0418 | True |

Canh bao:
- C2 khong dat: spike (share_train=1.16%, min_n=0)
- C2 khong dat: cold_anomaly (share_train=8.27%, min_n=219)
- C4 khong dat: spike (lech tuong doi val=0.21, test=1.00, nguong 0.5)
- C4 khong dat: heat_anomaly (lech tuong doi val=0.35, test=0.93, nguong 0.5)
- C4 khong dat: cold_anomaly (lech tuong doi val=0.67, test=0.74, nguong 0.5)

### 7. Uu tien co dinh va diem z lon nhat

So cua so hoa z (ca chuoi, hoa thi giu luat dung truoc trong priority): 0

| n_z_ties_all_windows | split | n | n_changed | pct_changed | n_multi_fire | n_changed_multi | pct_changed_multi |
|---|---|---|---|---|---|---|---|
| 0 | train | 48981 | 1153 | 0.0235 | 2455 | 1153 | 0.4697 |
| 0 | val | 10377 | 167 | 0.0161 | 389 | 167 | 0.4293 |
| 0 | test | 10377 | 216 | 0.0208 | 449 | 216 | 0.4811 |

## Cach B - dac trung tho

Thu tu uu tien: spike > pressure_drop > humidity_rise > heat_anomaly > cold_anomaly > normal

### 3. Nguong va ty le kich hoat (truoc uu tien)

| rule | feature | op | quantile | threshold |
|---|---|---|---|---|
| spike | T__jump__raw | >= | 0.9900 | 8.2400 |
| pressure_drop | p__delta__raw | <= | 0.1000 | -12.1900 |
| humidity_rise | rh__delta__raw | >= | 0.9000 | 20.6900 |
| heat_anomaly | T__mean__raw | >= | 0.9000 | 18.8220 |
| cold_anomaly | T__mean__raw | <= | 0.1000 | -1.6303 |

| rule | share_train | share_val | share_test |
|---|---|---|---|
| spike | 0.0116 | 0.0092 | 0.0000 |
| pressure_drop | 0.1001 | 0.1028 | 0.0904 |
| humidity_rise | 0.1000 | 0.0919 | 0.0808 |
| heat_anomaly | 0.1000 | 0.1173 | 0.1137 |
| cold_anomaly | 0.1000 | 0.0141 | 0.0265 |

### 4. Ma tran chong lan tren train (so cua so; duong cheo = so cua so cua luat)

| rule | spike | pressure_drop | humidity_rise | heat_anomaly | cold_anomaly |
|---|---|---|---|---|---|
| spike | 570 | 49 | 89 | 120 | 0 |
| pressure_drop | 49 | 4901 | 531 | 220 | 726 |
| humidity_rise | 89 | 531 | 4899 | 767 | 157 |
| heat_anomaly | 120 | 220 | 767 | 4900 | 0 |
| cold_anomaly | 0 | 726 | 157 | 0 | 4899 |

P(cot | hang) tren train:

| rule | spike | pressure_drop | humidity_rise | heat_anomaly | cold_anomaly |
|---|---|---|---|---|---|
| spike | 1.000 | 0.086 | 0.156 | 0.211 | 0.000 |
| pressure_drop | 0.010 | 1.000 | 0.108 | 0.045 | 0.148 |
| humidity_rise | 0.018 | 0.108 | 1.000 | 0.157 | 0.032 |
| heat_anomaly | 0.024 | 0.045 | 0.157 | 1.000 | 0.000 |
| cold_anomaly | 0.000 | 0.148 | 0.032 | 0.000 | 1.000 |

### 5. Nhan sau uu tien, va tieu chi C2-C4

| label | n_train | share_train | n_val | share_val | n_test | share_test |
|---|---|---|---|---|---|---|
| spike | 570 | 0.0116 | 95 | 0.0092 | 0 | 0.0000 |
| pressure_drop | 4852 | 0.0991 | 1067 | 0.1028 | 938 | 0.0904 |
| humidity_rise | 4282 | 0.0874 | 876 | 0.0844 | 789 | 0.0760 |
| heat_anomaly | 3880 | 0.0792 | 962 | 0.0927 | 1036 | 0.0998 |
| cold_anomaly | 4038 | 0.0824 | 125 | 0.0120 | 250 | 0.0241 |
| normal | 31359 | 0.6402 | 7252 | 0.6989 | 7364 | 0.7096 |

| label | share_train | n_train | n_val | n_test | min_n | share>=2% | share>=5% | min_n>=300 | C2 |
|---|---|---|---|---|---|---|---|---|---|
| spike | 0.0116 | 570 | 95 | 0 | 0 | False | False | False | False |
| pressure_drop | 0.0991 | 4852 | 1067 | 938 | 938 | True | True | True | True |
| humidity_rise | 0.0874 | 4282 | 876 | 789 | 789 | True | True | True | True |
| heat_anomaly | 0.0792 | 3880 | 962 | 1036 | 962 | True | True | True | True |
| cold_anomaly | 0.0824 | 4038 | 125 | 250 | 125 | True | True | False | False |

| normal_train | normal_val | normal_test | all<=70% | all<=60% |
|---|---|---|---|---|
| 0.6402 | 0.6989 | 0.7096 | False | False |

| label | share_train | share_val | share_test | rel_dev_val | rel_dev_test | C4 |
|---|---|---|---|---|---|---|
| spike | 0.0116 | 0.0092 | 0.0000 | 0.2133 | 1.0000 | False |
| pressure_drop | 0.0991 | 0.1028 | 0.0904 | 0.0380 | 0.0875 | True |
| humidity_rise | 0.0874 | 0.0844 | 0.0760 | 0.0344 | 0.1303 | True |
| heat_anomaly | 0.0792 | 0.0927 | 0.0998 | 0.1703 | 0.2603 | True |
| cold_anomaly | 0.0824 | 0.0120 | 0.0241 | 0.8539 | 0.7078 | False |
| normal | 0.6402 | 0.6989 | 0.7096 | 0.0916 | 0.1084 | True |

Canh bao:
- C2 khong dat: spike (share_train=1.16%, min_n=0)
- C2 khong dat: cold_anomaly (share_train=8.24%, min_n=125)
- C3 khong dat: Normal = 64.0% / 69.9% / 71.0% (train/val/test), nguong 70%
- C4 khong dat: spike (lech tuong doi val=0.21, test=1.00, nguong 0.5)
- C4 khong dat: cold_anomaly (lech tuong doi val=0.85, test=0.71, nguong 0.5)

### 7. Uu tien co dinh va diem z lon nhat

So cua so hoa z (ca chuoi, hoa thi giu luat dung truoc trong priority): 0

| n_z_ties_all_windows | split | n | n_changed | pct_changed | n_multi_fire | n_changed_multi | pct_changed_multi |
|---|---|---|---|---|---|---|---|
| 0 | train | 48981 | 1078 | 0.0220 | 2435 | 1078 | 0.4427 |
| 0 | val | 10377 | 102 | 0.0098 | 334 | 102 | 0.3054 |
| 0 | test | 10377 | 95 | 0.0092 | 218 | 95 | 0.4358 |

## 6. Cold Front - chi dem, khong gan nhan

Luat khai bao `front_dir`: ap suat doi chieu (hai nua cua so trai dau va min(|d1|,|d2|) >= phan vi 0.75 cua chinh dai luong do tren moi cua so train) VA delta T <= phan vi 0.10 VA |delta wd| vong dau-cuoi >= phan vi 0.90.
d1 = p(mid) - p(dau), d2 = p(cuoi) - p(mid), mid = L//2.
Cach A dung chuoi p/T da tru khi hau. Cach B dung chuoi tho. Huong gio khong tru khi hau.
`front_speed`, `front_either_wind`, `front_dir_fall_rise` la so dem cua dinh nghia gio / chieu ap khac, cung nguong. Bang khong xep hang chung.

Nguong train:

| mode | tag | q_turn | thr_leg | q_cool | thr_t_delta | q_wind | thr_wd | thr_wv |
|---|---|---|---|---|---|---|---|---|
| A | resid | 0.7500 | 5.3700 | 0.1000 | -7.0098 | 0.9000 | 160.0500 | 3.2174 |
| B | raw | 0.7500 | 5.3500 | 0.1000 | -6.8100 | 0.9000 | 160.0500 | 3.2300 |

So cua so:

| mode | component | n_train | share_train | n_val | share_val | n_test | share_test |
|---|---|---|---|---|---|---|---|
| A | pressure_reversal | 8877 | 0.1812 | 1763 | 0.1699 | 1655 | 0.1595 |
| A | pressure_fall_then_rise | 4848 | 0.0990 | 969 | 0.0934 | 838 | 0.0808 |
| A | rapid_cooling | 4899 | 0.1000 | 865 | 0.0834 | 957 | 0.0922 |
| A | wind_dir | 4899 | 0.1000 | 1124 | 0.1083 | 1032 | 0.0995 |
| A | wind_speed | 4899 | 0.1000 | 1054 | 0.1016 | 1189 | 0.1146 |
| A | front_dir | 101 | 0.0021 | 21 | 0.0020 | 23 | 0.0022 |
| A | front_dir_fall_rise | 53 | 0.0011 | 8 | 0.0008 | 18 | 0.0017 |
| A | front_speed | 95 | 0.0019 | 19 | 0.0018 | 28 | 0.0027 |
| A | front_either_wind | 181 | 0.0037 | 33 | 0.0032 | 49 | 0.0047 |
| B | pressure_reversal | 8862 | 0.1809 | 1775 | 0.1711 | 1653 | 0.1593 |
| B | pressure_fall_then_rise | 4840 | 0.0988 | 976 | 0.0941 | 847 | 0.0816 |
| B | rapid_cooling | 4899 | 0.1000 | 883 | 0.0851 | 1017 | 0.0980 |
| B | wind_dir | 4899 | 0.1000 | 1124 | 0.1083 | 1032 | 0.0995 |
| B | wind_speed | 4905 | 0.1001 | 1060 | 0.1021 | 1177 | 0.1134 |
| B | front_dir | 101 | 0.0021 | 21 | 0.0020 | 21 | 0.0020 |
| B | front_dir_fall_rise | 48 | 0.0010 | 7 | 0.0007 | 15 | 0.0014 |
| B | front_speed | 104 | 0.0021 | 16 | 0.0015 | 35 | 0.0034 |
| B | front_either_wind | 191 | 0.0039 | 31 | 0.0030 | 55 | 0.0053 |

## 8. Do nhay nguong

Moi o la d_share_train: ty le nhan tren train sau khi dich, tru ty le goc.
`kind=rule`: chi dich phan vi cua mot luat. `kind=all`: dich q_spike, q_low va q_high cung mot luc (q_low gan vao ca pressure_drop va cold_anomaly; q_high gan vao ca humidity_rise va heat_anomaly).
Phan vi bi cat ve [0, 1] thi clipped = True.

| mode | kind | rule | delta_pp | clipped | spike | pressure_drop | humidity_rise | heat_anomaly | cold_anomaly | normal |
|---|---|---|---|---|---|---|---|---|---|---|
| A | all | all | -2 | False | 0.0194 | -0.0215 | 0.0171 | 0.0126 | -0.0154 | -0.0123 |
| A | all | all | -1 | False | 0.0097 | -0.0103 | 0.0081 | 0.0073 | -0.0077 | -0.0072 |
| A | all | all | 1 | False | -0.0097 | 0.0100 | -0.0101 | -0.0082 | 0.0077 | 0.0104 |
| A | all | all | 2 | True | -0.0097 | 0.0199 | -0.0199 | -0.0161 | 0.0154 | 0.0104 |
| A | rule | cold_anomaly | -2 | False | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0165 | 0.0165 |
| A | rule | cold_anomaly | -1 | False | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0082 | 0.0082 |
| A | rule | cold_anomaly | 1 | False | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0081 | -0.0081 |
| A | rule | cold_anomaly | 2 | False | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0165 | -0.0165 |
| A | rule | heat_anomaly | -2 | False | 0.0000 | 0.0000 | 0.0000 | 0.0158 | 0.0000 | -0.0158 |
| A | rule | heat_anomaly | -1 | False | 0.0000 | 0.0000 | 0.0000 | 0.0082 | 0.0000 | -0.0082 |
| A | rule | heat_anomaly | 1 | False | 0.0000 | 0.0000 | 0.0000 | -0.0084 | 0.0000 | 0.0084 |
| A | rule | heat_anomaly | 2 | False | 0.0000 | 0.0000 | 0.0000 | -0.0163 | 0.0000 | 0.0163 |
| A | rule | humidity_rise | -2 | False | 0.0000 | 0.0000 | 0.0175 | -0.0022 | -0.0011 | -0.0142 |
| A | rule | humidity_rise | -1 | False | 0.0000 | 0.0000 | 0.0089 | -0.0011 | -0.0006 | -0.0072 |
| A | rule | humidity_rise | 1 | False | 0.0000 | 0.0000 | -0.0092 | 0.0011 | 0.0006 | 0.0075 |
| A | rule | humidity_rise | 2 | False | 0.0000 | 0.0000 | -0.0181 | 0.0022 | 0.0011 | 0.0148 |
| A | rule | pressure_drop | -2 | False | 0.0000 | -0.0197 | 0.0027 | 0.0019 | 0.0028 | 0.0123 |
| A | rule | pressure_drop | -1 | False | 0.0000 | -0.0099 | 0.0015 | 0.0010 | 0.0013 | 0.0061 |
| A | rule | pressure_drop | 1 | False | 0.0000 | 0.0100 | -0.0012 | -0.0010 | -0.0010 | -0.0068 |
| A | rule | pressure_drop | 2 | False | 0.0000 | 0.0199 | -0.0023 | -0.0021 | -0.0023 | -0.0133 |
| A | rule | spike | -2 | False | 0.0194 | -0.0019 | -0.0031 | -0.0022 | 0.0000 | -0.0122 |
| A | rule | spike | -1 | False | 0.0097 | -0.0004 | -0.0022 | -0.0005 | 0.0000 | -0.0066 |
| A | rule | spike | 1 | False | -0.0097 | 0.0000 | 0.0002 | 0.0004 | 0.0001 | 0.0090 |
| A | rule | spike | 2 | True | -0.0097 | 0.0000 | 0.0002 | 0.0004 | 0.0001 | 0.0090 |
| B | all | all | -2 | False | 0.0191 | -0.0207 | 0.0194 | 0.0066 | -0.0146 | -0.0098 |
| B | all | all | -1 | False | 0.0096 | -0.0104 | 0.0102 | 0.0040 | -0.0073 | -0.0061 |
| B | all | all | 1 | False | -0.0097 | 0.0105 | -0.0090 | -0.0064 | 0.0079 | 0.0066 |
| B | all | all | 2 | True | -0.0097 | 0.0203 | -0.0188 | -0.0148 | 0.0153 | 0.0078 |
| B | rule | cold_anomaly | -2 | False | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0163 | 0.0163 |
| B | rule | cold_anomaly | -1 | False | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0079 | 0.0079 |
| B | rule | cold_anomaly | 1 | False | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0085 | -0.0085 |
| B | rule | cold_anomaly | 2 | False | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0165 | -0.0165 |
| B | rule | heat_anomaly | -2 | False | 0.0000 | 0.0000 | 0.0000 | 0.0167 | 0.0000 | -0.0167 |
| B | rule | heat_anomaly | -1 | False | 0.0000 | 0.0000 | 0.0000 | 0.0085 | 0.0000 | -0.0085 |
| B | rule | heat_anomaly | 1 | False | 0.0000 | 0.0000 | 0.0000 | -0.0082 | 0.0000 | 0.0082 |
| B | rule | heat_anomaly | 2 | False | 0.0000 | 0.0000 | 0.0000 | -0.0168 | 0.0000 | 0.0168 |
| B | rule | humidity_rise | -2 | False | 0.0000 | 0.0000 | 0.0181 | -0.0024 | -0.0009 | -0.0148 |
| B | rule | humidity_rise | -1 | False | 0.0000 | 0.0000 | 0.0090 | -0.0011 | -0.0005 | -0.0074 |
| B | rule | humidity_rise | 1 | False | 0.0000 | 0.0000 | -0.0090 | 0.0011 | 0.0004 | 0.0074 |
| B | rule | humidity_rise | 2 | False | 0.0000 | 0.0000 | -0.0180 | 0.0025 | 0.0008 | 0.0147 |
| B | rule | pressure_drop | -2 | False | 0.0000 | -0.0197 | 0.0029 | 0.0011 | 0.0025 | 0.0132 |
| B | rule | pressure_drop | -1 | False | 0.0000 | -0.0098 | 0.0016 | 0.0004 | 0.0011 | 0.0068 |
| B | rule | pressure_drop | 1 | False | 0.0000 | 0.0101 | -0.0012 | -0.0010 | -0.0009 | -0.0070 |
| B | rule | pressure_drop | 2 | False | 0.0000 | 0.0198 | -0.0023 | -0.0019 | -0.0017 | -0.0138 |
| B | rule | spike | -2 | False | 0.0191 | -0.0013 | -0.0017 | -0.0081 | 0.0000 | -0.0080 |
| B | rule | spike | -1 | False | 0.0096 | -0.0008 | -0.0004 | -0.0034 | 0.0000 | -0.0051 |
| B | rule | spike | 1 | False | -0.0097 | 0.0004 | 0.0011 | 0.0019 | 0.0000 | 0.0063 |
| B | rule | spike | 2 | True | -0.0097 | 0.0004 | 0.0011 | 0.0019 | 0.0000 | 0.0063 |

Day du train/val/test: `threshold_sensitivity.csv`.

## 9. Mau cua so tren train

Moi nhan lay toi da 8 cua so train, seed 0, khong lap, ve theo thu tu thoi gian.
Don vi vat ly, 4 kenh. Anh: `figures/samples_<A|B>_<nhan>.png`. Chi so: `sample_indices.csv`.

| mode | label | n_plotted |
|---|---|---|
| A | cold_anomaly | 8 |
| A | heat_anomaly | 8 |
| A | humidity_rise | 8 |
| A | normal | 8 |
| A | pressure_drop | 8 |
| A | spike | 8 |
| B | cold_anomaly | 8 |
| B | heat_anomaly | 8 |
| B | humidity_rise | 8 |
| B | normal | 8 |
| B | pressure_drop | 8 |
| B | spike | 8 |

## 10. Do dai cum cua so lien tiep cung nhan

Tinh trong tung tap sau khi gan theo uu tien co dinh. Buoc cua so = 1 gio. Purge gap nam giua cac tap nen khong tao them mot dut cum ben trong tap.
mean_run_length = so cua so trung binh cua mot cum. n_windows / n_runs la do dai trung binh khi gop moi nhan.

Gop moi nhan:

| mode | split | n_windows | n_runs | mean_run_length | median_run_length |
|---|---|---|---|---|---|
| A | train | 48981 | 2241 | 21.8568 | 7.0000 |
| A | val | 10377 | 461 | 22.5098 | 8.0000 |
| A | test | 10377 | 438 | 23.6918 | 7.0000 |
| B | train | 48981 | 2239 | 21.8763 | 7.0000 |
| B | val | 10377 | 463 | 22.4125 | 8.0000 |
| B | test | 10377 | 441 | 23.5306 | 7.0000 |

Theo nhan:

| mode | split | label | n_windows | n_runs | mean_run_length | median_run_length | max_run_length |
|---|---|---|---|---|---|---|---|
| A | train | spike | 570 | 6 | 95.0000 | 95.0000 | 95 |
| A | train | pressure_drop | 4870 | 185 | 26.3243 | 19.0000 | 97 |
| A | train | humidity_rise | 4336 | 871 | 4.9782 | 3.0000 | 40 |
| A | train | heat_anomaly | 3702 | 173 | 21.3988 | 13.0000 | 181 |
| A | train | cold_anomaly | 4049 | 110 | 36.8091 | 19.0000 | 286 |
| A | train | normal | 31454 | 896 | 35.1049 | 15.0000 | 273 |
| A | val | spike | 95 | 1 | 95.0000 | 95.0000 | 95 |
| A | val | pressure_drop | 1071 | 36 | 29.7500 | 21.0000 | 102 |
| A | val | humidity_rise | 862 | 186 | 4.6344 | 3.0000 | 33 |
| A | val | heat_anomaly | 1058 | 32 | 33.0625 | 13.5000 | 192 |
| A | val | cold_anomaly | 279 | 12 | 23.2500 | 11.5000 | 105 |
| A | val | normal | 7012 | 194 | 36.1443 | 17.0000 | 377 |
| A | test | spike | 0 | 0 | nan | nan | 0 |
| A | test | pressure_drop | 929 | 36 | 25.8056 | 16.5000 | 111 |
| A | test | humidity_rise | 776 | 171 | 4.5380 | 2.0000 | 24 |
| A | test | heat_anomaly | 1511 | 48 | 31.4792 | 14.0000 | 228 |
| A | test | cold_anomaly | 219 | 6 | 36.5000 | 33.5000 | 63 |
| A | test | normal | 6942 | 177 | 39.2203 | 18.0000 | 298 |
| B | train | spike | 570 | 6 | 95.0000 | 95.0000 | 95 |
| B | train | pressure_drop | 4852 | 186 | 26.0860 | 19.5000 | 96 |
| B | train | humidity_rise | 4282 | 902 | 4.7472 | 3.0000 | 39 |
| B | train | heat_anomaly | 3880 | 177 | 21.9209 | 12.0000 | 131 |
| B | train | cold_anomaly | 4038 | 77 | 52.4416 | 27.0000 | 271 |
| B | train | normal | 31359 | 891 | 35.1953 | 17.0000 | 313 |
| B | val | spike | 95 | 1 | 95.0000 | 95.0000 | 95 |
| B | val | pressure_drop | 1067 | 34 | 31.3824 | 21.5000 | 102 |
| B | val | humidity_rise | 876 | 190 | 4.6105 | 3.0000 | 33 |
| B | val | heat_anomaly | 962 | 32 | 30.0625 | 16.0000 | 161 |
| B | val | cold_anomaly | 125 | 6 | 20.8333 | 18.5000 | 49 |
| B | val | normal | 7252 | 200 | 36.2600 | 17.5000 | 255 |
| B | test | spike | 0 | 0 | nan | nan | 0 |
| B | test | pressure_drop | 938 | 37 | 25.3514 | 17.0000 | 106 |
| B | test | humidity_rise | 789 | 175 | 4.5086 | 3.0000 | 24 |
| B | test | heat_anomaly | 1036 | 34 | 30.4706 | 17.0000 | 161 |
| B | test | cold_anomaly | 250 | 4 | 62.5000 | 52.5000 | 144 |
| B | test | normal | 7364 | 191 | 38.5550 | 18.0000 | 368 |

## Canh bao C2-C4

- [A] C2 khong dat: spike (share_train=1.16%, min_n=0)
- [A] C2 khong dat: cold_anomaly (share_train=8.27%, min_n=219)
- [A] C4 khong dat: spike (lech tuong doi val=0.21, test=1.00, nguong 0.5)
- [A] C4 khong dat: heat_anomaly (lech tuong doi val=0.35, test=0.93, nguong 0.5)
- [A] C4 khong dat: cold_anomaly (lech tuong doi val=0.67, test=0.74, nguong 0.5)
- [B] C2 khong dat: spike (share_train=1.16%, min_n=0)
- [B] C2 khong dat: cold_anomaly (share_train=8.24%, min_n=125)
- [B] C3 khong dat: Normal = 64.0% / 69.9% / 71.0% (train/val/test), nguong 70%
- [B] C4 khong dat: spike (lech tuong doi val=0.21, test=1.00, nguong 0.5)
- [B] C4 khong dat: cold_anomaly (lech tuong doi val=0.85, test=0.71, nguong 0.5)

