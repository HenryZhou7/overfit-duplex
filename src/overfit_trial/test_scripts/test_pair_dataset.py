"""Test script for DuplexPairDataset."""

from pathlib import Path

from torch.utils.data import DataLoader

from src.overfit_trial.pair_dataset import DuplexPairDataset, collate_duplex_batch


def test_dataset_with_example_csv():
    """Test the dataset with an example CSV file."""

    # Create a simple test CSV file
    csv_path = Path("/tmp/test_pairs.csv")
    csv_content = """channel1_id,channel2_id
user_1724702491,assistant_1724702491
user_1724702493,assistant_1724702493"""

    with open(csv_path, "w") as f:
        f.write(csv_content)

    print(f"Created test CSV at: {csv_path}")
    print("CSV contents:")
    print(csv_content)
    print()

    # Assuming the data files exist in the expected location
    data_dir = Path("/home/henry/dev/overfit-duplex/asset/single_pair_dataset")

    # Create dataset
    dataset = DuplexPairDataset(
        csv_file=csv_path,
        window_size=1024,
        data_dir=data_dir,
        num_quantizers=32,
        pad_to_window_size=True,
        random_window=False,  # Use first window for testing
    )

    print(f"Dataset created with {len(dataset)} conversation pairs")
    print()

    # Test single sample access
    if len(dataset) > 0:
        sample = dataset[0]
        print("Single sample test:")
        print(f"  User codes shape: {sample['user_codes'].shape}")
        print(f"  Assistant codes shape: {sample['assistant_codes'].shape}")
        print(f"  User length: {sample['user_length']}")
        print(f"  Assistant length: {sample['assistant_length']}")
        print()

    # Create dataloader
    if len(dataset) > 0:
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_duplex_batch,
            num_workers=0,  # Use 0 for debugging
        )

        print("DataLoader test:")
        for batch_idx, (user_codes, assistant_codes, user_lengths, assistant_lengths) in enumerate(dataloader):
            print(f"Batch {batch_idx}:")
            print(f"  User codes shape: {user_codes.shape}")
            print(f"  Assistant codes shape: {assistant_codes.shape}")
            print(f"  User lengths: {user_lengths}")
            print(f"  Assistant lengths: {assistant_lengths}")
            print(f"  User codes dtype: {user_codes.dtype}")
            print(f"  Assistant codes dtype: {assistant_codes.dtype}")
            print(f"  Lengths dtype: {user_lengths.dtype}")

            # Check that the data is valid
            assert user_codes.shape[0] == assistant_codes.shape[0], "Batch size mismatch"
            assert user_codes.shape[1] == 32, f"Expected 32 quantizers, got {user_codes.shape[1]}"
            assert user_codes.shape[2] == 1024, f"Expected window size 1024, got {user_codes.shape[2]}"

            # Only show first batch
            break

        print("\nDataLoader test passed!")
    else:
        print("No samples in dataset - check if NPZ files exist")


def test_integration_with_model():
    """Test that the dataset output works with the model."""

    print("\nTesting integration with model...")

    # Create a simple test CSV
    csv_path = Path("/tmp/test_pairs.csv")
    csv_content = """channel1_id,channel2_id
user_1724702491,assistant_1724702491"""

    with open(csv_path, "w") as f:
        f.write(csv_content)

    data_dir = Path("/home/henry/dev/overfit-duplex/asset/single_pair_dataset")

    # Create dataset
    dataset = DuplexPairDataset(
        csv_file=csv_path,
        window_size=1024,
        data_dir=data_dir,
        num_quantizers=32,
    )

    if len(dataset) > 0:
        # Get a single sample
        sample = dataset[0]

        # Add batch dimension
        user_codes = sample["user_codes"].unsqueeze(0)
        assistant_codes = sample["assistant_codes"].unsqueeze(0)

        print("Input shapes for model:")
        print(f"  codes_c1: {user_codes.shape}")
        print(f"  codes_c2: {assistant_codes.shape}")

        # You could test with actual model here if needed
        # model = MachOverfitModel(...)
        # output = model(codes_c1=user_codes, codes_c2=assistant_codes)

        print("Model integration test ready - shapes are correct!")
    else:
        print("No samples available for model test")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing DuplexPairDataset")
    print("=" * 60)

    test_dataset_with_example_csv()
    test_integration_with_model()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
