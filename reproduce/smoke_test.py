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

    # Disease-area split smoke check: requires torch_geometric (PyG).
    # Import here to surface missing dependency early and explicitly.
    from torch_geometric.utils import k_hop_subgraph  # noqa: F401
    tx_data.prepare_split(
        split="anemia",
        seed=42,
        test_size=0.01,
        one_hop=True,
        mask_ratio=0.01,
    )

    print("Smoke test passed: model initialized and disease-area split ran.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
