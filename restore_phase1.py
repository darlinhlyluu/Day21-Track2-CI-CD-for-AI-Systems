import pandas as pd


def main():
    train_path = "data/train_phase1.csv"
    phase2_path = "data/train_phase2.csv"

    df_train = pd.read_csv(train_path)
    df_phase2 = pd.read_csv(phase2_path)

    phase2_size = len(df_phase2)
    expected_combined_size = phase2_size * 2

    if len(df_train) == phase2_size:
        print(
            "train_phase1.csv da o trang thai goc (2998 mau). "
            "Khong can khoi phuc."
        )
        return

    if len(df_train) != expected_combined_size:
        raise ValueError(
            "Khong the khoi phuc tu dong vi train_phase1.csv khong co kich thuoc "
            f"du kien. Hien tai: {len(df_train)} dong du lieu, mong doi: "
            f"{expected_combined_size}."
        )

    if not df_train.tail(phase2_size).reset_index(drop=True).equals(
        df_phase2.reset_index(drop=True)
    ):
        raise ValueError(
            "Khong the khoi phuc tu dong vi phan cuoi cua train_phase1.csv "
            "khong khop voi train_phase2.csv."
        )

    df_original = df_train.head(phase2_size)
    df_original.to_csv(train_path, index=False)

    print(
        f"Khoi phuc thanh cong train_phase1.csv: {len(df_train)} -> "
        f"{len(df_original)} mau"
    )


if __name__ == "__main__":
    main()
