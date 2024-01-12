"""prepare the data for training """ 
def dataPrepare():
    # load data from file
    from aa_readfile import Dataread
    file_path = 'AMLS_23_24_SN23071850/Datasets/pneumoniamnist.npz'
    dataset=Dataread(file_path)

    #load each category as a variable
    from aa_readfile import category_Data
    train_images,train_labels,val_images,val_labels,test_images,test_labels=category_Data(dataset)

    # random show selected image
    from ab_sample_image import display_random_images
    display_random_images(train_images, train_labels, num_images=5)

    # enhance the data to expand dataset
    from ac_enhance_image import enhanced_data
    augmented_train_images, augmented_train_labels=enhanced_data(train_images, train_labels)

    # normalise the data for pre-process
    from ad_Normalisation import normalisation
    train_images_normalized,augmented_train_images_normalized,val_images_normalized,test_images_normalized=normalisation(train_images,val_images,test_images,augmented_train_images)

    return train_images_normalized,augmented_train_images_normalized,val_images_normalized,test_images_normalized,train_labels,val_labels,test_labels,augmented_train_labels