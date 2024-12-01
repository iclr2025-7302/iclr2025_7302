from huggingface_hub import hf_hub_download

def download_test_file():
    hf_hub_download(
        repo_id="ICLR2025-7302/CIFAR10-890k", 
        filename="cifar10_factor_graph.jls",
        local_dir=".",
        local_dir_use_symlinks=False
    )

if __name__ == "__main__":
    download_test_file()
