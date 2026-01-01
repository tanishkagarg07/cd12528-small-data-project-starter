def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Paths
    # -----------------------------
    ORIGINAL_DATA_PATH = "data/loan_continuous.csv"
    DENIED_ONLY_PATH = "data/loan_denied_only.csv"
    AUGMENTED_PATH = "data/loan_continuous_expanded.csv"

    # -----------------------------
    # Load original data
    # -----------------------------
    df = pd.read_csv(ORIGINAL_DATA_PATH)

    # Split out Loan_Status = 1 (Denied)
    denied_df = df[df["Loan_Status"] == 1]
    denied_df.to_csv(DENIED_ONLY_PATH, index=False)

    # -----------------------------
    # Baseline model performance
    # -----------------------------
    print("Baseline Model Performance")
    test_model(ORIGINAL_DATA_PATH)

    # -----------------------------
    # Create Datasets & Loaders
    # -----------------------------
    train_dataset = DataBuilder(DENIED_ONLY_PATH, train=True)
    val_dataset = DataBuilder(DENIED_ONLY_PATH, train=False)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # -----------------------------
    # Initialize VAE
    # -----------------------------
    input_dim = train_dataset.x.shape[1]
    model = Autoencoder(D_in=input_dim).to(device)
    criterion = CustomLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # -----------------------------
    # Train & Validate
    # -----------------------------
    epochs = 1000
    model.train()

    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0

        # Training
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss = criterion(recon, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon, mu, logvar = model(batch)
                loss = criterion(recon, batch, mu, logvar)
                val_loss += loss.item()

        model.train()

        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Train Loss: {train_loss:.2f} | Val Loss: {val_loss:.2f}")

    # -----------------------------
    # Generate Synthetic Data
    # -----------------------------
    scaler = train_dataset.standardizer
    fake_data = generate_fake(mu, logvar, 50000, scaler, model)

    fake_df = pd.DataFrame(fake_data, columns=denied_df.columns)
    fake_df["Loan_Status"] = 1

    # -----------------------------
    # Combine Real + Synthetic
    # -----------------------------
    augmented_df = pd.concat([df, fake_df], ignore_index=True)
    augmented_df.to_csv(AUGMENTED_PATH, index=False)

    # -----------------------------
    # Test Augmented Dataset
    # -----------------------------
    print("\nAfter Synthetic Data Augmentation")
    test_model(AUGMENTED_PATH)
