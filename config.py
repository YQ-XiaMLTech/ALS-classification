

class Config:
    main_path = "./"
    dataset = "dataset/AptamerROIs020623"
    label = "dataset/ROI Image Key.xlsx"
    classification = 'multi-classification'
    # 'multi-classification','binary-classification'
    type_classification = 'Control'+'Concordant'+'Discordant'

    # 'Control'+'Concordant', 'Control'+'Concordant'+'Discordant', 'Concordant'+'Discordant','Control'+'Discordant'
    # model="GNEW"
    # max_natoms = 648
    # length = 190
    # process_dst_path = '/Users/xiayuqing/Documents/senior/Graduate project/code/ChemGNN-master/processed/GCN/GCN_C1P'
    train_ratio = 0.6
    test_ratio = 0.2
    val_ratio = 0.2
    # root_bmat = main_path + 'data/{}/BTMATRIXES'.format(dataset)
    # root_dmat = main_path + 'data/{}/DMATRIXES'.format(dataset)
    # root_conf = main_path + 'data/{}/CONFIG'.format(dataset)
    # root_force = main_path + 'data/{}/FORCE'.format(dataset)
    # format_bmat = "BTMATRIX_{}"
    # format_dmat = "DTMATRIX_{}"
    # format_conf = "water{}"
    # format_force = "MLFORCE_{}"
    # format_eigen = "MLENERGY_{}"
    #format_charge = "CHARGES/charge_data_{}"
    # loss_fn_id = 1
    # threshold = -17.2
    # tblog_dir = "logs"
    model = "ResNet18"

    epoch = 10
    # epoch_step = 1  # print loss every {epoch_step} epochs
    batch_size= 64
    lr = 0.00013
    seed = 1
config = Config