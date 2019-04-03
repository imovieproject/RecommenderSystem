from surprise import KNNBaseline,KNNBasic
from surprise import Dataset,Reader

import pandas as pd

class RFIbcf(KNNBaseline):
    """ 使用RF-ICBF算法的推荐引擎类,训练数据在构造函数时传入 """

    # 训练数据数据帧
    __train_df=None
    # 训练数据数据集
    # TODO 删除这个属性
    __trainset=None
    # 所有用户评分过的电影列表
    __rated_movie_rids={}
    # 所有用户未评分过的电影列表
    __unrated_movie_rids={}

    def __init__(self,train_df,k=40,min_k=1,sim_options={},bsl_options={}):
        """ 类构造函数
        Args:
            train_df: 训练数据的数据帧
        """
        self.__train_df=train_df

        # 保证是协同过滤是基于物品的
        if 'user_based' not in sim_options.keys():
            sim_options['user_based']=False
        KNNBaseline.__init__(self,k=k,min_k=min_k,sim_options=sim_options,bsl_options=bsl_options)


    def __get_rating_frequency(self):
        """ 计算训练数据中用户和物品的评分频率
        Returns:
            user_rating_frequency: 用户评分频率字典，key为用户rid，value为用户各分值的评分次数及总的评分平均分
            movie_rating_frequency: 物品评分频率字典，key为物品rid，value为物品各分值的评分次数及总的评分平均分
        """
        user_rating_frequency={}
        movie_rating_frequency={}
        count=0
        for row in self.__train_df.iterrows():
            print("processing record {}".format(count))
            count+=1

            userid=int(row[1]['userId'])
            movieid=int(row[1]['movieId'])

            if userid not in user_rating_frequency.keys():
                # 一共有十种可能的评分
                # 第一个元素表示0.5分出现的频率,依次类推
                user_rating_frequency[userid]={}
                user_rating_frequency[userid]['list']=[0]*10
                user_rating_frequency[userid]['mean']=0

            if movieid not in movie_rating_frequency.keys():
                movie_rating_frequency[movieid]={}
                movie_rating_frequency[movieid]['list']=[0]*10
                movie_rating_frequency[movieid]['mean']=0

            user_rating_frequency[userid]['list'][int(row[1]['rating']/0.5-1)]+=1
            movie_rating_frequency[movieid]['list'][int(row[1]['rating']/0.5-1)]+=1
            user_rating_frequency[userid]['mean']+= row[1]['rating']
            movie_rating_frequency[movieid]['mean']+=row[1]['rating']
        
        # 计算每个用户的均值
        for key in user_rating_frequency.keys():
            print('calculationg mean of user {}'.format(key))
            user_rating_frequency[key]['mean']=user_rating_frequency[key]['mean']/sum(user_rating_frequency[key]['list'])
            user_rating_frequency[key]['mean']=round(user_rating_frequency[key]['mean']*2)/2

        for key in movie_rating_frequency.keys():
            print('calculationg mean of movie {}'.format(key))
            movie_rating_frequency[key]['mean']=movie_rating_frequency[key]['mean']/sum(movie_rating_frequency[key]['list'])
            movie_rating_frequency[key]['mean']=round(movie_rating_frequency[key]['mean']*2)/2
        return user_rating_frequency,movie_rating_frequency

    def __predict_rating(self,user_rid,movie_rid,user_rating_info,movie_rating_info):
        """ 预测某个用户对某部电影的评分
        Args:
            user_rid: 用户rid
            movie_rid: 物品rid
            user_rating_info: 用户评分信息,包括各分值评分频率和历史评分均值
            movie_rating_info: 物品评分信息，包括各分值评分频率和历史评分均值
        Returns:
            max_rating: 预测的评分
        """
        # 最大的预测值
        max=0
        # 使max达到最大的分值
        max_rating=0
        # 评分频率列表
        user_rating_frequency_list=user_rating_info['list']
        movie_rating_frequency_list=movie_rating_info['list']
        # 评分均值
        user_rating_mean=user_rating_info['mean']
        movie_rating_mean=movie_rating_info['mean']

        for i in range(0,10):
                rating=0.5*(i+1)
                predict_rating=(user_rating_frequency_list[i]+1+(rating==user_rating_mean))*(movie_rating_frequency_list[i]+1+(rating==movie_rating_mean))
                if predict_rating>max:
                    max=predict_rating
                    max_rating=rating

        return max_rating

    def __preprocess_trainset(self,thresholdu,thresholdi):
        """ 使用RF-REC方法预测训练集中的评分 
        Args:
            thresholdu: 用户评分频率的阈值
            thresholdi: 物品评分频率的阈值
        """
        user_rating_frequency,movie_rating_frequency=self.__get_rating_frequency()

        # 将训练集按用户rid分组
        groups=self.__train_df.groupby['userId'].unique()
        # 获取训练集里面所有的rid列表
        movie_rids=self.__train_df['movieId'].unique()
        # 遍历用户组，预测用户未评分过的评分
        for user_rid,group in groups:
            if sum(user_rating_frequency[user_rid]['list'])>thresholdu:
                self.__rated_movie_rids[user_rid]=group['movieId'].unique()
                self.__unrated_movie_rids[user_rid]=list(set(movie_rids)-set(self.__rated_movie_rids))
                for movie_rid in self.__unrated_movie_rids:
                    if sum(movie_rating_frequency[movie_rid]['list'])>thresholdi:
                        predicted_rating=self.__predict_rating(
                                                           user_rid=user_rid,
                                                           movie_rid=movie_rid,
                                                           user_rating_info=user_rating_frequency[user_rid],
                                                           movie_rating_info=movie_rating_frequency[movie_rid] 
                        )
                        self.__train_df=self.__train_df.append(pd.DataFrame(
                            [[user_rid,movie_rid,predicted_rating,0]],
                            columns=['userId','movieId','rating','timestamp']
                        ))

        # 将添加评分预测后的Dataframe按userId、movieId排序
        self.__train_df=self.__train_df.sort_values(by=['userId','movieId'])

    def load_trainset_df(self,trainset_df):
        """ 加载训练数据集数据帧 
        Args:
            trainset_df: DataFrame对象或者训练数据集的路径
        """
        if isinstance(trainset_df,str):
            self.__train_df=pd.read_csv(trainset_df)
        elif isinstance(trainset_df,DataFrame):
            self.__train_df=trainset_df 

    def estimate(self,u,i):
        KNNBaseline.estimate(self,u,i)

    def fit(self,trainset=None):
        # 对数据进行处理
        self.__preprocess_trainset(thresholdu=190,thresholdi=190)
        # 读取处理后的训练集数据帧
        reader=Reader(rating_scale(1,5))
        data=DataFrame.load_from_df(self.__train_df[['userId','movieId','rating']],reader=reader)
        trainset=data.build_full_trainset()

        # 调用父类的fit函数
        KNNBaseline.fit(self,trainset)
        return self        

    def reommend(self,N=10):
        """ 根据计算得到的相似度进行推荐 
        Args:
            N: 推荐结果数
        Return：
            recommend_list: 包含N个元素，每个元素是又包含推荐物品rid和用户感兴趣程度
        """
        recommend_list={}

        # 获取所有用户和物品rid
        user_rids=self.__train_df['userId'].unique()
        movie_rids=self.__train_df['movieId'].unique()

        # 对每个用户都生成TopN推荐
        for user_rid in user_rids:
            recommend_list[user_rid]=[]
            # 获取用户评价过的电影rid
            rated_movies=self.__rated_movie_rids[user_rid]
            # 获取与评分过的电影相似的电影
            for movie_rid in rated_movies:
                movie_iid=self.trainset.to_inner_iid(movie_rid)
                k_neighbors_iid=algo.get_neighbors(inner_id,k=10)
                k_neighbors_rid=[self.trainset.to_raw_iid(iid) for iid in k_neighbors_iid]

                # 删除用户已经评价过的电影
                k_neighbors_rid=list(set(k_neighbors_rid)-set(rated_movies))

                # 计算每部推荐电影的用户感兴趣程度
                for i in range(0,len(k_neighbors_rid)):
                    # 用户对电影的评分
                    rating=self.__train_df[
                        (self.__train_df.userId==user_rid)&
                        (self.__train_df.movieId==movie_rid)
                        ].loc[0,'rating']
                    recommend_list[user_rid].append(
                        [
                            user_rid,
                            self.sim[movie_iid][k_neighbors_iid[i]]*rating
                        ]
                    )

        # 将得到的推荐结果按感兴趣程度排序
        sorted(recommend_list[user_rid],key=lambda x: x[1],reverse=True)
        # 截取前n项
        recommend_list[user_rid]=recommend_list[user_rid][0:N]

    # 测试代码
    def get_rating_frequency(self):
        return self.__get_rating_frequency()

    def get_train_df(self):
        return self.__train_df

    def get_trainset(self):
        return self.__trainset

    def test_predict_rating(self):
        user_rid=1
        movie_rid=1
        user_rating_info={
            'list':[0,2,0,0,0,0,0,1,0,1],
            'mean': 3
        }
        movie_rating_info={
            'list':[0,2,0,0,0,0,0,0,0,10],
            'mean': 2
        }

        print(self.__predict_rating(user_rid,movie_rid,user_rating_info,movie_rating_info))


test=1.0
if __name__ == '__main__':
    trainset=pd.read_csv('ratings.csv')
    algo=RFIbcf(trainset)
    algo.test_predict_rating()