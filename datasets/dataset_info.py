import numpy as np


class DatasetInfo():

    def __init__(self):
        self.name = ''
        self.self_collected = False
        self.data_dir = ''
        self.train_submap_type = ''
        self.test_submap_type = ''
        self.cloud_ext = ''
        self.cloud_dtype = np.float64
        self.train_trip_names = None  # trips in training dataset
        self.test_trip_names = None  # trips in testing dataset
        self.test_region_vertices = []
        self.test_region_width = 0.0
        self.test_query_trips = None  # used it when computing recall/precision, | None means all! |
        self.train_radius_pos = 0.0
        self.train_radius_neg = 0.0
        self.test_radius_pos = 0.0
        self.test_radius_neg = 0.0
        self.global_offset = np.array([[0.0, 0.0, 0.0]])

    def train_cfg(self):
        return {
            'name': self.name,
            'is_test_dataset': False,
            'self_collected': self.self_collected,
            'data_dir': self.data_dir,
            'submap_type': self.train_submap_type,
            'cloud_ext': self.cloud_ext,
            'cloud_dtype': self.cloud_dtype,
            'trip_names': self.train_trip_names,
            'test_region_vertices': self.test_region_vertices,
            'test_region_width': self.test_region_width,
            'test_query_trips': None,
            'search_radius_pos': self.train_radius_pos,
            'search_radius_neg': self.train_radius_neg,
            'skip_trip_itself': False,
            'global_offset': self.global_offset
        }

    def test_cfg(self):
        return {
            'name': self.name,
            'is_test_dataset': True,
            'self_collected': self.self_collected,
            'data_dir': self.data_dir,
            'submap_type': self.test_submap_type,
            'cloud_ext': self.cloud_ext,
            'cloud_dtype': self.cloud_dtype,
            'trip_names': self.test_trip_names,
            'test_region_vertices': self.test_region_vertices,
            'test_region_width': self.test_region_width,
            'test_query_trips': self.test_query_trips,
            'search_radius_pos': self.test_radius_pos,
            'search_radius_neg': self.test_radius_neg,
            'skip_trip_itself': True,
            'global_offset': self.global_offset
        }

# data info dict
dataset_info_dict = dict()

# WHU Dataset: hankou
hankou_info = DatasetInfo()
hankou_info.name = 'hankou'
hankou_info.self_collected = True
hankou_info.data_dir = '/home/ericxhzou/Code/benchmark_datasets/wh_hankou_origin'
hankou_info.train_submap_type = 'pointcloud_30m_2m_clean'
hankou_info.test_submap_type = 'pointcloud_30m_2m_clean'
hankou_info.cloud_ext = '.bin'
hankou_info.test_region_vertices = []
hankou_info.test_region_width = 50.0
hankou_info.test_query_trips = ['helmet_submap']
hankou_info.train_radius_pos = 15.0
hankou_info.train_radius_neg = 60.0
hankou_info.test_radius_pos = 30.0
hankou_info.test_radius_neg = 60.0
hankou_info.global_offset = np.array([[794000.0, 385000.0, 30.0]])
dataset_info_dict[hankou_info.name] = hankou_info

# WHU Dataset: campus
campus_info = DatasetInfo()
campus_info.name = 'campus'
campus_info.self_collected = True
campus_info.data_dir = '/home/ericxhzou/Code/benchmark_datasets/whu_campus_origin'
campus_info.train_submap_type = 'pointcloud_30m_2m_clean'
campus_info.test_submap_type = 'pointcloud_30m_2m_clean'
campus_info.cloud_ext = '.bin'
campus_info.test_region_vertices = [[0.0, 0.0]]
campus_info.test_region_width = 1.e10
campus_info.test_query_trips = ['helmet_submap']
campus_info.train_radius_pos = 15.0
campus_info.train_radius_neg = 60.0
campus_info.test_radius_pos = 30.0
campus_info.test_radius_neg = 60.0
campus_info.global_offset = np.array([[534000.0, 3379000.0, 25.0]])
dataset_info_dict[campus_info.name] = campus_info

# WHU Dataset: campus - for training rerank-net
campus_rerank_info = DatasetInfo()
campus_rerank_info.name = 'campus_rerank'
campus_rerank_info.self_collected = True
campus_rerank_info.data_dir = '/home/ericxhzou/DataDisk/benchmark_datasets/whu_campus_rerank'
campus_rerank_info.train_submap_type = 'pointcloud_30m_2m_clean'
campus_rerank_info.test_submap_type = 'pointcloud_30m_2m_clean'
campus_rerank_info.cloud_ext = '.bin'
campus_rerank_info.test_region_vertices = []  # empty means all for training
campus_rerank_info.test_region_width = 50.0
campus_rerank_info.test_query_trips = ['helmet_submap']
campus_rerank_info.train_radius_pos = 15.0
campus_rerank_info.train_radius_neg = 60.0
campus_rerank_info.test_radius_pos = 30.0
campus_rerank_info.test_radius_neg = 60.0
dataset_info_dict[campus_rerank_info.name] = campus_rerank_info

# Oxford RobotCar Dataset: only for place recognition
oxford_info = DatasetInfo()
oxford_info.name = 'oxford'
oxford_info.self_collected = False
oxford_info.data_dir = '/home/ericxhzou/DataDisk/benchmark_datasets/oxford'
oxford_info.train_submap_type = 'pointcloud_20m_10overlap'
oxford_info.test_submap_type = 'pointcloud_20m'
oxford_info.cloud_ext = '.bin'
oxford_info.test_trip_names = ['2014-11-14-16-34-33', '2014-11-18-13-20-12', '2014-12-02-15-30-08', '2014-12-09-13-21-02',
                               '2014-12-10-18-10-50', '2014-12-12-10-45-15', '2014-12-16-09-14-09', '2014-12-16-18-44-24',
                               '2015-02-03-08-45-10', '2015-02-10-11-58-05', '2015-02-13-09-16-26', '2015-02-17-14-42-12',
                               '2015-03-10-14-18-10', '2015-03-17-11-08-44', '2015-05-19-14-06-38', '2015-06-09-15-06-29',
                               '2015-08-12-15-04-18', '2015-08-13-16-02-58', '2015-08-14-14-54-57', '2015-08-28-09-50-22',
                               '2015-10-30-13-52-14', '2015-11-12-11-22-05', '2015-11-13-10-28-08']
oxford_info.test_region_vertices = [[5735712.768124, 620084.402381],
                                    [5735611.299219, 620540.270327],
                                    [5735237.358209, 620543.094379],
                                    [5734749.303802, 619932.693364]]
oxford_info.test_region_width = 150.0
oxford_info.test_query_trips = None  # all sequences are for query
oxford_info.train_radius_pos = 10.0
oxford_info.train_radius_neg = 50.0
oxford_info.test_radius_pos = 25.0
oxford_info.test_radius_neg = 50.0
dataset_info_dict[oxford_info.name] = oxford_info

# 3-Inhouse Dataset: university
university_info = DatasetInfo()
university_info.name = 'university'
university_info.self_collected = False
university_info.data_dir = '/home/ericxhzou/DataDisk/benchmark_datasets/inhouse_datasets/university'
university_info.train_submap_type = 'pointcloud_25m_25'
university_info.test_submap_type = 'pointcloud_25m_25'
university_info.cloud_ext = '.bin'
university_info.test_region_vertices = [[363621.292362, 142864.19756],
                                        [364788.795462, 143125.746609],
                                        [363597.507711, 144011.414174]]
university_info.test_region_width = 150.0
university_info.test_query_trips = None
university_info.train_radius_pos = 12.5
university_info.train_radius_neg = 50.0
university_info.test_radius_pos = 25.0
university_info.test_radius_neg = 50.0
dataset_info_dict[university_info.name] = university_info

# 3-Inhouse Dataset: residential
residential_info = DatasetInfo()
residential_info.name = 'residential'
residential_info.self_collected = False
residential_info.data_dir = '/home/ericxhzou/DataDisk/benchmark_datasets/inhouse_datasets/residential'
residential_info.train_submap_type = 'pointcloud_25m_25'
residential_info.test_submap_type = 'pointcloud_25m_25'
residential_info.cloud_ext = '.bin'
residential_info.test_region_vertices = [[360895.486453, 144999.915143],
                                         [362357.024536, 144894.825301],
                                         [361368.907155, 145209.663042]]
residential_info.test_region_width = 150.0
residential_info.test_query_trips = None
residential_info.train_radius_pos = 12.5
residential_info.train_radius_neg = 50.0
residential_info.test_radius_pos = 25.0
residential_info.test_radius_neg = 50.0
dataset_info_dict[residential_info.name] = residential_info

# 3-Inhouse Dataset: business
business_info = DatasetInfo()
business_info.name = 'business'
business_info.self_collected = False
business_info.data_dir = '/home/ericxhzou/DataDisk/benchmark_datasets/inhouse_datasets/business'
business_info.train_submap_type = 'pointcloud_25m_25'
business_info.test_submap_type = 'pointcloud_25m_25'
business_info.cloud_ext = '.bin'
business_info.test_region_vertices = [[0.0, 0.0]]
business_info.test_region_width = 1.e10
business_info.test_query_trips = None
business_info.train_radius_pos = 12.5
business_info.train_radius_neg = 50.0
business_info.test_radius_pos = 25.0
business_info.test_radius_neg = 50.0
dataset_info_dict[business_info.name] = business_info

# MulRan Dataset: sejong, all for training!
sejong_info = DatasetInfo()
sejong_info.name = 'sejong'
sejong_info.self_collected = False
sejong_info.data_dir = '/home/ericxhzou/DataDisk/benchmark_datasets/MulRan/TrainDataSejong01_02'
sejong_info.train_submap_type = 'pointcloud_0.2m'
sejong_info.test_submap_type = 'pointcloud_0.2m'
sejong_info.cloud_ext = '.bin'
sejong_info.cloud_dtype = np.float32
sejong_info.test_region_vertices = []
sejong_info.test_region_width = 50.0
sejong_info.test_query_trips = None
sejong_info.train_radius_pos = 2.0
sejong_info.train_radius_neg = 10.0
sejong_info.test_radius_pos = 20.0  # or 5.0
sejong_info.test_radius_neg = 20.0  # FIXME: undefined in EgoNN
dataset_info_dict[sejong_info.name] = sejong_info

# MulRan Dataset: dcc -> step 20m, all for testing!
dcc20_info = DatasetInfo()
dcc20_info.name = 'dcc_20m'
dcc20_info.self_collected = False
dcc20_info.data_dir = '/home/ericxhzou/DataDisk/benchmark_datasets/MulRan/TestDataDCC01_02'
dcc20_info.train_submap_type = 'pointcloud_10.0m'
dcc20_info.test_submap_type = 'pointcloud_10.0m'
dcc20_info.cloud_ext = '.bin'
dcc20_info.cloud_dtype = np.float32
dcc20_info.test_region_vertices = [[0.0, 0.0]]
dcc20_info.test_region_width = 1.e10
dcc20_info.test_query_trips = ['query_seq1']
dcc20_info.train_radius_pos = 2.0  # FIXME: undefined in EgoNN
dcc20_info.train_radius_neg = 10.0  # FIXME: undefined in EgoNN
dcc20_info.test_radius_pos = 20.0
dcc20_info.test_radius_neg = 20.0  # FIXME: undefined in EgoNN
dataset_info_dict[dcc20_info.name] = dcc20_info

# MulRan Dataset: dcc -> step 5m, all for testing!
dcc5_info = DatasetInfo()
dcc5_info.name = 'dcc_5m'
dcc5_info.self_collected = False
dcc5_info.data_dir = '/home/ericxhzou/DataDisk/benchmark_datasets/MulRan/TestDataDCC01_02_step5m'
dcc5_info.train_submap_type = 'pointcloud_10.0m'
dcc5_info.test_submap_type = 'pointcloud_10.0m'
dcc5_info.cloud_ext = '.bin'
dcc5_info.cloud_dtype = np.float32
dcc5_info.test_region_vertices = [[0.0, 0.0]]
dcc5_info.test_region_width = 1.e10
dcc5_info.test_query_trips = ['query_seq1']
dcc5_info.train_radius_pos = 2.0  # FIXME: undefined in EgoNN
dcc5_info.train_radius_neg = 10.0  # FIXME: undefined in EgoNN
dcc5_info.test_radius_pos = 5.0
dcc5_info.test_radius_neg = 20.0  # FIXME: undefined in EgoNN
dataset_info_dict[dcc5_info.name] = dcc5_info

# KITTI-360 Dataset: seq 09 -> step 20m
kitti360_20m_info = DatasetInfo()
kitti360_20m_info.name = 'kitti360_20m'
kitti360_20m_info.self_collected = False
kitti360_20m_info.data_dir = '/home/ericxhzou/DataDisk/benchmark_datasets/KITTI_360/TestDataSeq09'
kitti360_20m_info.train_submap_type = 'pointcloud_3.0m'
kitti360_20m_info.test_submap_type = 'pointcloud_3.0m'
kitti360_20m_info.cloud_ext = '.bin'
kitti360_20m_info.cloud_dtype = np.float32
kitti360_20m_info.test_region_vertices = [[0.0, 0.0]]
kitti360_20m_info.test_region_width = 1.e10
kitti360_20m_info.test_query_trips = ['query_seq1']
kitti360_20m_info.train_radius_pos = 2.0  # FIXME: undefined in EgoNN
kitti360_20m_info.train_radius_neg = 10.0  # FIXME: undefined in EgoNN
kitti360_20m_info.test_radius_pos = 20.0
kitti360_20m_info.test_radius_neg = 20.0  # FIXME: undefined in EgoNN
dataset_info_dict[kitti360_20m_info.name] = kitti360_20m_info

# KITTI-360 Dataset: seq 09 -> step 5m
kitti360_5m_info = DatasetInfo()
kitti360_5m_info.name = 'kitti360_5m'
kitti360_5m_info.self_collected = False
kitti360_5m_info.data_dir = '/home/ericxhzou/DataDisk/benchmark_datasets/KITTI_360/TestDataSeq09_step5m'
kitti360_5m_info.train_submap_type = 'pointcloud_3.0m'
kitti360_5m_info.test_submap_type = 'pointcloud_3.0m'
kitti360_5m_info.cloud_ext = '.bin'
kitti360_5m_info.cloud_dtype = np.float32
kitti360_5m_info.test_region_vertices = [[0.0, 0.0]]
kitti360_5m_info.test_region_width = 1.e10
kitti360_5m_info.test_query_trips = ['query_seq1']
kitti360_5m_info.train_radius_pos = 2.0  # FIXME: undefined in EgoNN
kitti360_5m_info.train_radius_neg = 10.0  # FIXME: undefined in EgoNN
kitti360_5m_info.test_radius_pos = 5.0
kitti360_5m_info.test_radius_neg = 20.0  # FIXME: undefined in EgoNN
dataset_info_dict[kitti360_5m_info.name] = kitti360_5m_info