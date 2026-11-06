# Pattern stats - Jena (buoc 2: nhan thu, chong lan, tieu chi C2-C4)

Chi do, khong chon thay nguoi dung. Nguong chi tinh tren train; nhan chi dung dac trung cua X.

- Cua so: train 48981, val 10377, test 10377 (tong 69973, purge 119)
- Phan vi: spike 0.99, thap 0.1, cao 0.9; kenh spike: T
- Cach A: dac trung phan du; anom = mean(x - khi hau). Cach B: dac trung tho; heat/cold dung trung binh tho cua T (khong tru khi hau).
- C2: share_train >= 2% va so cua so nho nhat qua 3 tap >= 300; C3: Normal <= 70% o ca 3 tap; C4: do lech tuong doi so voi train <= 0.5 (dung sai C4 la mac dinh cua script, prompt khong neu).

## Cach A - dac trung phan du so voi khi hau

Thu tu uu tien: spike > pressure_drop > humidity_rise > heat_anomaly > cold_anomaly > normal

Nguong (chi tu train):

| rule | feature | op | quantile | threshold |
|---|---|---|---|---|
| spike | T__jump__resid | >= | 0.9900 | 7.6911 |
| pressure_drop | p__delta__resid | <= | 0.1000 | -12.2190 |
| humidity_rise | rh__delta__resid | >= | 0.9000 | 20.4280 |
| heat_anomaly | T__anom | >= | 0.9000 | 4.7405 |
| cold_anomaly | T__anom | <= | 0.1000 | -4.6102 |

Ty le moi luat kich hoat (truoc uu tien, moi luat doc lap):

| rule | share_train | share_val | share_test |
|---|---|---|---|
| spike | 0.0116 | 0.0092 | 0.0000 |
| pressure_drop | 0.1000 | 0.1032 | 0.0895 |
| humidity_rise | 0.1000 | 0.0916 | 0.0799 |
| heat_anomaly | 0.1000 | 0.1245 | 0.1813 |
| cold_anomaly | 0.1000 | 0.0340 | 0.0241 |

Nhan sau khi gan theo uu tien:

| label | n_train | share_train | n_val | share_val | n_test | share_test |
|---|---|---|---|---|---|---|
| spike | 570 | 0.0116 | 95 | 0.0092 | 0 | 0.0000 |
| pressure_drop | 4870 | 0.0994 | 1071 | 0.1032 | 929 | 0.0895 |
| humidity_rise | 4336 | 0.0885 | 862 | 0.0831 | 776 | 0.0748 |
| heat_anomaly | 3702 | 0.0756 | 1058 | 0.1020 | 1511 | 0.1456 |
| cold_anomaly | 4049 | 0.0827 | 279 | 0.0269 | 219 | 0.0211 |
| normal | 31454 | 0.6422 | 7012 | 0.6757 | 6942 | 0.6690 |

Chong lan, so cua so train thoa dong thoi (duong cheo = so cua so cua luat):

| rule | spike | pressure_drop | humidity_rise | heat_anomaly | cold_anomaly |
|---|---|---|---|---|---|
| spike | 570 | 29 | 42 | 94 | 3 |
| pressure_drop | 29 | 4899 | 521 | 600 | 591 |
| humidity_rise | 42 | 521 | 4899 | 642 | 301 |
| heat_anomaly | 94 | 600 | 642 | 4899 | 0 |
| cold_anomaly | 3 | 591 | 301 | 0 | 4899 |

Xac suat co dieu kien P(cot | hang) tren train:

| rule | spike | pressure_drop | humidity_rise | heat_anomaly | cold_anomaly |
|---|---|---|---|---|---|
| spike | 1.000 | 0.051 | 0.074 | 0.165 | 0.005 |
| pressure_drop | 0.006 | 1.000 | 0.106 | 0.122 | 0.121 |
| humidity_rise | 0.009 | 0.106 | 1.000 | 0.131 | 0.061 |
| heat_anomaly | 0.019 | 0.122 | 0.131 | 1.000 | 0.000 |
| cold_anomaly | 0.001 | 0.121 | 0.061 | 0.000 | 1.000 |

### Tieu chi C2 (moi nhan khac Normal)

| label | share_train | n_train | n_val | n_test | min_n | share>=2% | share>=5% | min_n>=300 | C2 |
|---|---|---|---|---|---|---|---|---|---|
| spike | 0.0116 | 570 | 95 | 0 | 0 | False | False | False | False |
| pressure_drop | 0.0994 | 4870 | 1071 | 929 | 929 | True | True | True | True |
| humidity_rise | 0.0885 | 4336 | 862 | 776 | 776 | True | True | True | True |
| heat_anomaly | 0.0756 | 3702 | 1058 | 1511 | 1058 | True | True | True | True |
| cold_anomaly | 0.0827 | 4049 | 279 | 219 | 219 | True | True | False | False |

### Tieu chi C3 (ty le Normal)

| normal_train | normal_val | normal_test | all<=70% | all<=60% |
|---|---|---|---|---|
| 0.6422 | 0.6757 | 0.6690 | True | False |

### Tieu chi C4 (lech ty le giua cac tap)

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

## Cach B - dac trung tho (L boi chu ky ngay)

Thu tu uu tien: spike > pressure_drop > humidity_rise > heat_anomaly > cold_anomaly > normal

Nguong (chi tu train):

| rule | feature | op | quantile | threshold |
|---|---|---|---|---|
| spike | T__jump__raw | >= | 0.9900 | 8.2400 |
| pressure_drop | p__delta__raw | <= | 0.1000 | -12.1900 |
| humidity_rise | rh__delta__raw | >= | 0.9000 | 20.6900 |
| heat_anomaly | T__mean__raw | >= | 0.9000 | 18.8220 |
| cold_anomaly | T__mean__raw | <= | 0.1000 | -1.6303 |

Ty le moi luat kich hoat (truoc uu tien, moi luat doc lap):

| rule | share_train | share_val | share_test |
|---|---|---|---|
| spike | 0.0116 | 0.0092 | 0.0000 |
| pressure_drop | 0.1001 | 0.1028 | 0.0904 |
| humidity_rise | 0.1000 | 0.0919 | 0.0808 |
| heat_anomaly | 0.1000 | 0.1173 | 0.1137 |
| cold_anomaly | 0.1000 | 0.0141 | 0.0265 |

Nhan sau khi gan theo uu tien:

| label | n_train | share_train | n_val | share_val | n_test | share_test |
|---|---|---|---|---|---|---|
| spike | 570 | 0.0116 | 95 | 0.0092 | 0 | 0.0000 |
| pressure_drop | 4852 | 0.0991 | 1067 | 0.1028 | 938 | 0.0904 |
| humidity_rise | 4282 | 0.0874 | 876 | 0.0844 | 789 | 0.0760 |
| heat_anomaly | 3880 | 0.0792 | 962 | 0.0927 | 1036 | 0.0998 |
| cold_anomaly | 4038 | 0.0824 | 125 | 0.0120 | 250 | 0.0241 |
| normal | 31359 | 0.6402 | 7252 | 0.6989 | 7364 | 0.7096 |

Chong lan, so cua so train thoa dong thoi (duong cheo = so cua so cua luat):

| rule | spike | pressure_drop | humidity_rise | heat_anomaly | cold_anomaly |
|---|---|---|---|---|---|
| spike | 570 | 49 | 89 | 120 | 0 |
| pressure_drop | 49 | 4901 | 531 | 220 | 726 |
| humidity_rise | 89 | 531 | 4899 | 767 | 157 |
| heat_anomaly | 120 | 220 | 767 | 4900 | 0 |
| cold_anomaly | 0 | 726 | 157 | 0 | 4899 |

Xac suat co dieu kien P(cot | hang) tren train:

| rule | spike | pressure_drop | humidity_rise | heat_anomaly | cold_anomaly |
|---|---|---|---|---|---|
| spike | 1.000 | 0.086 | 0.156 | 0.211 | 0.000 |
| pressure_drop | 0.010 | 1.000 | 0.108 | 0.045 | 0.148 |
| humidity_rise | 0.018 | 0.108 | 1.000 | 0.157 | 0.032 |
| heat_anomaly | 0.024 | 0.045 | 0.157 | 1.000 | 0.000 |
| cold_anomaly | 0.000 | 0.148 | 0.032 | 0.000 | 1.000 |

### Tieu chi C2 (moi nhan khac Normal)

| label | share_train | n_train | n_val | n_test | min_n | share>=2% | share>=5% | min_n>=300 | C2 |
|---|---|---|---|---|---|---|---|---|---|
| spike | 0.0116 | 570 | 95 | 0 | 0 | False | False | False | False |
| pressure_drop | 0.0991 | 4852 | 1067 | 938 | 938 | True | True | True | True |
| humidity_rise | 0.0874 | 4282 | 876 | 789 | 789 | True | True | True | True |
| heat_anomaly | 0.0792 | 3880 | 962 | 1036 | 962 | True | True | True | True |
| cold_anomaly | 0.0824 | 4038 | 125 | 250 | 125 | True | True | False | False |

### Tieu chi C3 (ty le Normal)

| normal_train | normal_val | normal_test | all<=70% | all<=60% |
|---|---|---|---|---|
| 0.6402 | 0.6989 | 0.7096 | False | False |

### Tieu chi C4 (lech ty le giua cac tap)

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

