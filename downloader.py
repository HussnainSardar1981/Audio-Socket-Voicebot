
    # Initialize downloader
    downloader = MicrosoftGraphDownloader(
        client_id=client_id,
        client_secret=client_secret,
        tenant_id=tenant_id,
        sharepoint_site_url=sharepoint_site_url
    )

    # Download PDFs for all customers
    server_root = Path(server_root)

    # Optional: filter specific customers
    # customers = ["Stuart Dean", "Feinblum"]  # Only these customers
    customers = None  # All customers

    results = downloader.download_all_customers(
        server_root=server_root,
        customer_filter=customers,
        force_redownload=False  # Set to True to re-download all PDFs
    )

    # Exit with status
    failed = sum(1 for r in results.values() if r['status'] != 'success')
    sys.exit(1 if failed > 0 else 0)


if __name__ == '__main__':
    main()
