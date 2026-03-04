import os
import sys

from txgnn import TxData, TxGNN


def main():
    data_dir = os.environ.get("TXGNN_DATA", "./data")
    kg_path = os.path.join(data_dir, "kg.csv")

    if not os.path.exists(kg_path):
        print(
            f"Smoke test skipped: {kg_path} not found. Set TXGNN_DATA to an existing data folder."
        )
        return 0

    tx_data = TxData(data_folder_path=data_dir)
    tx_data.prepare_split(split="complex_disease", seed=42)

    tx_gnn = TxGNN(
        data=tx_data,
        weight_bias_track=False,
        proj_name="TxGNN",
        exp_name="TxGNN",
        device="cpu",
    )

    tx_gnn.model_initialize(
        n_hid=32,
        n_inp=32,
        n_out=32,
        proto=False,
        attention=False,
        sim_measure="all_nodes_profile",
        agg_measure="rarity",
        num_walks=10,
        path_length=2,
    )

    print("Smoke test passed: model initialized.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
