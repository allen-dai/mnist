use std::fs;
use std::io::{self, Read};
use std::path;

const IMAGE_SIZE: usize = 28 * 28;

pub struct MnistLoader {
    pub train_images: Vec<[u8; IMAGE_SIZE]>,
    pub test_images: Vec<[u8; IMAGE_SIZE]>,
    pub train_labels: Vec<u8>,
    pub test_labels: Vec<u8>,
}

impl MnistLoader {
    pub fn new<T: AsRef<path::Path>>(
        train_img_path: T,
        train_label_path: T,
        test_img_path: T,
        test_label_path: T,
    ) -> io::Result<Self> {
        /* Images
        -----------------------------------------------------------------------
        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000803(2051) magic number
        0004     32 bit integer  60000/10000      number of items
                                 train/test
        0008     32 bit integer  28               number of rows
        0012     32 bit integer  28               number of columns
        0016     unsigned byte   ??               pixel
        0017     unsigned byte   ??               pixel
        ........
        xxxx     unsigned byte   ??               pixel
        -----------------------------------------------------------------------
        */

        // Train images
        let (train_magic_num, train_num_img, train_row, train_col, train_images) =
            images(train_img_path)?;
        if train_magic_num != 2051 || train_num_img != 60000 || train_row != 28 || train_col != 28 {
            let invalid_error =
                io::Error::new(io::ErrorKind::InvalidData, "Incorrect train image data");
            return Err(invalid_error);
        }

        // Test images
        let (test_magic_num, test_num_img, test_row, test_col, test_images) =
            images(test_img_path)?;
        if test_magic_num != 2051 || test_num_img != 10000 || test_row != 28 || test_col != 28 {
            let invalid_error =
                io::Error::new(io::ErrorKind::InvalidData, "Incorrect test image data");
            return Err(invalid_error);
        }

        /* Labels
        -----------------------------------------------------------------------
        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000801(2049) magic number (MSB first)
        0004     32 bit integer  60000/10000      number of items
                                 train/test
        0008     unsigned byte   ??               label
        0009     unsigned byte   ??               label
        ........
        xxxx     unsigned byte   ??               label
        -----------------------------------------------------------------------
        */

        // Train labels
        let (train_label_magic_num, train_label_num_img, train_labels) = labels(train_label_path)?;
        if train_label_magic_num != 2049 || train_label_num_img != 60000 {
            let invalid_error =
                io::Error::new(io::ErrorKind::InvalidData, "Incorrect train label data");
            return Err(invalid_error);
        }

        // Test labels
        let (test_label_magic_num, test_label_num_img, test_labels) = labels(test_label_path)?;
        if test_label_magic_num != 2049 || test_label_num_img != 10000 {
            let invalid_error =
                io::Error::new(io::ErrorKind::InvalidData, "Incorrect test label data");
            return Err(invalid_error);
        }

        Ok(MnistLoader {
            train_images,
            train_labels,
            test_images,
            test_labels,
        })
    }
}

fn images(path: impl AsRef<path::Path>) -> io::Result<(u32, u32, u32, u32, Vec<[u8; IMAGE_SIZE]>)> {
    let mut file = fs::File::open(path)?;
    let mut buf: [u8; 4] = [0; 4];

    file.read_exact(&mut buf)?;
    let magic_number = u32::from_be_bytes(buf);
    file.read_exact(&mut buf)?;
    let num_images = u32::from_be_bytes(buf);
    file.read_exact(&mut buf)?;
    let num_rows = u32::from_be_bytes(buf);
    file.read_exact(&mut buf)?;
    let num_cols = u32::from_be_bytes(buf);

    let mut image_buf: [u8; IMAGE_SIZE] = [0; IMAGE_SIZE];
    let mut images = Vec::new();
    for _ in 0..num_images {
        file.by_ref()
            .take((num_rows * num_cols) as u64)
            .read(&mut image_buf)?;
        images.push(image_buf.clone());
    }
    Ok((magic_number, num_images, num_rows, num_cols, images))
}

fn labels(path: impl AsRef<path::Path>) -> io::Result<(u32, u32, Vec<u8>)> {
    let mut file = fs::File::open(path)?;
    let mut buf: [u8; 4] = [0; 4];

    file.read_exact(&mut buf)?;
    let magic_number = u32::from_be_bytes(buf);
    file.read_exact(&mut buf)?;
    let num_items = u32::from_be_bytes(buf);

    let mut label: [u8; 1] = [0];
    let mut labels = Vec::new();
    for _ in 0..num_items {
        file.by_ref().take(1).read(&mut label)?;
        labels.push(label[0]);
    }
    Ok((magic_number, num_items, labels))
}
