# 会议记录

## 2021年2月4日星期四

- 参与人员：牟永利，李斐斐，张泽宇，刘泉

- 会议议程
  - 进度管理
    - 已完成
      - GAP数据库图片的收集与识别
      - 建立Github共享文件
    - TODO
      - 基于GAP数据库完成模型训练和测试，给出初步结果报告
      - 收集路面检测视频片段
  - 团队管理
    - 明确每两周开一次会议（bi-weekly meeting）
    - 强调成果转化

- 会议计划

  - 会议时间：2021年2月20日星期六下午3点

  - 会议形式：[Zoom Meeting](https://rwth.zoom.us/j/93868590377?pwd=SFdITWpIbWUrWWQwcjlkWFFXcTN2QT09)

    Meeting ID: 938 6859 0377

    Passcode: 211719

Thanks Mou for your tutorial！！

## 2021年2月19日星期五

- 参与人员：牟永利，李斐斐
- 会议议程
  - 代码修改建议
    - Learning rate (Hyperparameter tuning)
    - Loss function ([KL DivLoss](https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html), Circle Loss, Wasserstein Loss)
    - Random Crop (选取crop中心点的位置时，根据是否为intact road改变被选中概率=>oversampling)
    - Normalization
    - Multi-view Training
    - Framework change: Unet/ Resnet34+Unet/SimpleClassifier+Unet or other segmentation networ


## 2021年2月20日星期六

- 参与人员：牟永利，李斐斐，张泽宇，刘泉，王宠惠
- 会议议程
   - 进展交流
      - 刘泉 85%精度
      - 张泽宇 硬件分析
      - 王宠惠 同一路段不同时段
   - TODO
      - 国内行业检查报告（刘泉）
      - 行业规范总结（刘泉）
      - 模型优化，算法优化 （牟永利，李斐斐）
      - 研究Meta learning对性能
   - 合作框架初步共识，涉及到资金流分布
   - 明确下阶段每人的任务及预计成果

## 2021年3月13日星期日

- 参与人员：牟永利，李斐斐，张泽宇，刘泉，王宠惠
- 会议议程
  - 进展交流
    - 李斐斐，牟永利
      - 总结实验结果
        - Class Imbalance问题
      - Publication
        - 基于统计的剪裁的重采样方法提高类别不平衡情况下模型训练效果
        - 基于强化学习模型的采样方法提高类别不平衡情况下模型训练效果
    - 刘泉
      - 汇报《国内路面病害检查报告》样式
    - 王宠惠
      - 
  - TODO
    - 李斐斐，牟永利
      - 完成实验和实验报告
    - 
      - 