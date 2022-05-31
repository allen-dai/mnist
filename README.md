# MNIST
A MNIST dataset parser, created for my own use.

# Usage
```rust

let loader = MnistLoader::new(train_img_path, test_img_path, train_label_path, test_label_path);

// Loader struct
pub struct MnistLoader {
    pub train_images: Vec<[u8; IMAGE_SIZE]>,
    pub test_images: Vec<[u8; IMAGE_SIZE]>,
    pub train_labels: Vec<u8>,
    pub test_labels: Vec<u8>,
}
```
