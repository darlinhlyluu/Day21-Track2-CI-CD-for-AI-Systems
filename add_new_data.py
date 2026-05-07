import pandas as pd

df_train = pd.read_csv("data/train_phase1.csv")
df_new   = pd.read_csv("data/train_phase2.csv")

original_size = len(df_train)

if original_size == len(df_new):
    df_updated = pd.concat([df_train, df_new], ignore_index=True)
    df_updated.to_csv("data/train_phase1.csv", index=False)
    print(f"Cap nhat du lieu: {original_size} -> {len(df_updated)} mau")
elif original_size == len(df_new) * 2 and df_train.tail(len(df_new)).reset_index(drop=True).equals(
    df_new.reset_index(drop=True)
):
    print(
        "train_phase1.csv da bao gom train_phase2.csv truoc do. "
        "Khong them du lieu lap."
    )
else:
    raise ValueError(
        "Khong the xac dinh trang thai du lieu hien tai de them train_phase2 "
        "mot cach an toan."
    )
