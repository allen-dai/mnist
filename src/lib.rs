extern crate byteorder;
use byteorder::{BigEndian, ReadBytesExt};
use std::fs;
use std::io::{self, Read};
use std::path;

const TRAIN_IMG_URL: &str = &"http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz";
const TRAIN_LBL_URL: &str = &"http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz";
const TEST_IMG_URL: &str = &"http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz";
const TEST_LBL_URL: &str = &"http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz";

pub struct Mnist {
    pub train_images: Vec<u8>,
    pub test_images: Vec<u8>,
    pub train_labels: Vec<u8>,
    pub test_labels: Vec<u8>,
}

impl Mnist {
    pub fn from_file<T: AsRef<path::Path>>(
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

        Ok(Mnist {
            train_images,
            train_labels,
            test_images,
            test_labels,
        })
    }

    pub fn from_download() -> anyhow::Result<Self> {
        use flate2::read::GzDecoder;
        use std::io::prelude::*;
        use std::io::BufWriter;
        fs::create_dir_all("./mnist_dataset")?;
        let train_img_path = std::path::Path::new("./mnist_dataset/train-images-idx3-ubyte.gz");
        let train_lbl_path = std::path::Path::new("./mnist_dataset/train-labels-idx1-ubyte.gz");
        let test_img_path = std::path::Path::new("./mnist_dataset/t10k-images-idx3-ubyte.gz");
        let test_lbl_path = std::path::Path::new("./mnist_dataset/t10k-labels-idx1-ubyte.gz");

        if !(train_img_path.exists()
            && train_lbl_path.exists()
            && test_img_path.exists()
            && test_lbl_path.exists())
        {
            println!("Downloading dataset from 'yann.lecun.com'...");
            let mut train_img_gz_file = std::fs::File::create(&train_img_path)?;
            let mut train_lbl_gz_file = std::fs::File::create(&train_lbl_path)?;
            let mut test_img_gz_file = std::fs::File::create(&test_img_path)?;
            let mut test_lbl_gz_file = std::fs::File::create(&test_lbl_path)?;

            reqwest::blocking::get(TRAIN_IMG_URL)?.copy_to(&mut train_img_gz_file)?;
            reqwest::blocking::get(TRAIN_LBL_URL)?.copy_to(&mut train_lbl_gz_file)?;
            reqwest::blocking::get(TEST_IMG_URL)?.copy_to(&mut test_img_gz_file)?;
            reqwest::blocking::get(TEST_LBL_URL)?.copy_to(&mut test_lbl_gz_file)?;
            println!("Finished downloading...");
        } else {
            println!("Found dataste...");
        }

        if !(train_img_path.with_extension("").exists()
            && train_lbl_path.with_extension("").exists()
            && test_img_path.with_extension("").exists()
            && test_lbl_path.with_extension("").exists())
        {
            println!("Decompressing dataset files...");
            let mut train_img_gz = GzDecoder::new(fs::File::open(train_img_path)?);
            let mut train_lbl_gz = GzDecoder::new(fs::File::open(train_lbl_path)?);
            let mut test_img_gz = GzDecoder::new(fs::File::open(test_img_path)?);
            let mut test_lbl_gz = GzDecoder::new(fs::File::open(test_lbl_path)?);

            let mut train_img_file =
                BufWriter::new(fs::File::create(train_img_path.with_extension(""))?);
            let mut train_lbl_file =
                BufWriter::new(fs::File::create(train_lbl_path.with_extension(""))?);
            let mut test_img_file =
                BufWriter::new(fs::File::create(test_img_path.with_extension(""))?);
            let mut test_lbl_file =
                BufWriter::new(fs::File::create(test_lbl_path.with_extension(""))?);

            // the largest file is train img file, which is ~10MB
            let mut buf: Vec<u8> = Vec::with_capacity(10 * (1024 * 1024));

            train_img_gz.read_to_end(&mut buf)?;
            train_img_file.write_all(&buf)?;
            buf.clear();

            train_lbl_gz.read_to_end(&mut buf)?;
            train_lbl_file.write_all(&buf)?;
            buf.clear();

            test_img_gz.read_to_end(&mut buf)?;
            test_img_file.write_all(&buf)?;
            buf.clear();

            test_lbl_gz.read_to_end(&mut buf)?;
            test_lbl_file.write_all(&buf)?;
            println!("Finished decompressing dataset files...");
        } else {
            println!("Found uncompressed file...");
        }
        println!("Loading up dataset...");
        Ok(Self::from_file(
            train_img_path.with_extension(""),
            train_lbl_path.with_extension(""),
            test_img_path.with_extension(""),
            test_lbl_path.with_extension(""),
        )?)
    }
}

fn images(path: impl AsRef<path::Path>) -> io::Result<(u32, u32, u32, u32, Vec<u8>)> {
    let mut content = Vec::new();
    let mut file = {
        let mut f = fs::File::open(path)?;
        let _ = f.read_to_end(&mut content)?;
        &content[..]
    };

    let magic_number = file.read_u32::<BigEndian>()?;
    let num_images = file.read_u32::<BigEndian>()?;
    let num_rows = file.read_u32::<BigEndian>()?;
    let num_cols = file.read_u32::<BigEndian>()?;
    Ok((magic_number, num_images, num_rows, num_cols, file.to_vec()))
}

fn labels(path: impl AsRef<path::Path>) -> io::Result<(u32, u32, Vec<u8>)> {
    let mut content = Vec::new();
    let mut file = {
        let mut f = fs::File::open(path)?;
        let _ = f.read_to_end(&mut content)?;
        &content[..]
    };

    let magic_number = file.read_u32::<BigEndian>()?;
    let num_items = file.read_u32::<BigEndian>()?;
    Ok((magic_number, num_items, file.to_vec()))
}

#[test]
fn test_download() {
    Mnist::from_download().unwrap();
}
