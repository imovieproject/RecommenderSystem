from surprise import AlgoBase
from surprise import Dataset,Reader

from recommender_engine.rf_ibcf import RFIbcf 

class Controller():
    """ 推荐引擎管理器 """
    # 推荐引擎列表
    # 需要保证list中的元素类型都是AlgoBase或A老公Base子类
    # TODO 设计类型检查机制
    __engines=list()
    __trainset_df=None
    # 最终推荐结果列表
    __recommend_list=list()

    def __init__(self,engines=None,trainset_df=None):
        self.__engines=engines
        self.__trainset=trainset_df


    def run():
        """ 运行所有的推荐引擎 
        """
        for engine in __engines:
            engine.run()
            recommend_list.append(engine.recommend_list)

    def get_recommend_list():
        return recommend_list
            