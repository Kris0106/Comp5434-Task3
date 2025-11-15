# Comp5434-Task3
数据处理，去掉重复值后，数据分类发现有num型，二元型，category型分别进行处理（StandardScaler，OneHotEncoder），其中发现avg_price_per_room和lead_time数据偏态 skewness>0（大部分是1，少数大数字），进行对数处理  

由于标签0/1比例差距太大，用scale_pos_weight去计算class weight 来address class imbalance  

先划分数据集train_test_split，然后对划分好的x_train再进行StratifiedKFold，并进行RandomizedSearchCV  

**消融**：用包裹法Wrapper Feature来优化（Feature Ablation）---筛出最小特征子集的  
过程：每次移除一个特征，且每次数据都重新处理，用最优超参数训练模型，如果性能提升，则移除那个特征并下一轮。



