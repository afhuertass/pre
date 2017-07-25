from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.lib.io.tf_record import TFRecordCompressionType
import tensorflow as tf

import pandas as pd
import numpy as np
from random import shuffle


MAX_LEN = 150

class DataParser(object):
    # convert data to tensorrecords
    
    def __init__(self, data_path ):
        # clase auxiliar
        self.product_path = data_path + "/products/products.csv"
        self.orders_path = data_path + "/orders/orders.csv"
        self.train_path = data_path +"/order_products__train/train.csv"
        self.prior_path = data_path + "/order_products__prior/prior.csv"
        

        self.num_products = 0 
        self.df_prior = None
        self.prepare_train_data()


    
    def prepare_train_data(self  ):
        reo = '1'
        #names=['order_id' , 'product_id' , 'add_to_cart_order'  , 'reordered' ]
        # ['order_id' , 'user_id' , 'eval_set'  ,'order_number' , 'order_dow' , 'order_hour_of_day' , 'days_since_prior_order']
        df_train = pd.read_csv( self.train_path , dtype= { 'order_id' : int , 'product_id' : int  , 'reordered' : int , 'add_to_cart_order' : int } )
        df_orders = pd.read_csv( self.orders_path, dtype = { 'order_id' : int , 'user_id' : int , 'eval_set' : str , 'order_number' : int , 'order_dow':int , 'order_hour_of_day': int , 'days_since_prior_order' : float  } )

        chunk =  pd.read_csv(self.prior_path , dtype = { 'order_id':int,'product_id':int,'add_to_cart_order':int,'reordered':int  }  )
        print("datasets-loaded")

        order_ids_train = df_train['order_id'].tolist()
        
        order_ids_train = list( set( order_ids_train ) )
        
        output = "./results/train_set.pb2"
        
        #df_prior = pd.read_csv( self.prior_path , dtype = { } )

        
        tot = len(order_ids_train)
        n = 0
        features_train = []
        features_target = []
        features_id = []
        shuffle( order_ids_train )
        for order_id_train in order_ids_train[:]:
            print(  order_id_train   )
            n = n +1 
            user_ids = df_orders[ df_orders['order_id'] == order_id_train   ]
            user_id = user_ids['user_id'].values[0]
            
            orders_id_prior = df_orders[ df_orders['user_id'] == user_id ]
            
            orders_id_prior = orders_id_prior[ orders_id_prior['eval_set'] == 'prior'  ]
            
            orders_id_prior =  orders_id_prior['order_id'].values
            

           
            #print("orders_prior")
            #print(orders_id_prior )
            products_train = []
            # ikariam1234 
            # kaggleun@gmail.com
            
            products_target = df_train[ df_train['order_id'] == order_id_train]
            products_target =  products_target[   products_target['reordered'] == 1  ]['product_id']
            products_target = products_target.tolist()
            
            
            for order_id in orders_id_prior:
                
               
                #for chunk in pd.read_csv(self.prior_path , dtype = { 'order_id':int,'product_id':int,'add_to_cart_order':int,'reordered':int  } , chunksize = size_c ):
                    # reading products 
                products = chunk[ chunk['order_id'] == order_id ]['product_id']
                products = products.values
                    
                #print( products )
                #print( products )
                for p in products:
                    products_train.append( p )

                # products_train = products previous
                #
            n_products = len( products_train )
            if n_products < MAX_LEN:
                rest =  MAX_LEN - n_products # we need to add rest products as ceros
                
                for i in np.arange(0 , rest):
                    products_train.append( 0)
            else:
                # keep just the last  MAX_LEN items
                products_train = products_train[-MAX_LEN:]
                
            #print( "products train" )
            #print( products_train )
            #print( "products target " )
            #print( products_target )
            print("Progress {}/{} , id processed {} ".format( n , tot , order_id_train  ) )
            
            
            features_train.append( products_train )
            features_target.append( products_target )
            features_id.append( order_id_train )
            
        self.dataset_tofile(self.instacar_feature( features_train , features_target , features_id ) , output )

            #return
        
        #self.dataset_tofile(self.instacar_feature( products_train , products_target ) , output )

                
    # xgboost
    
    def instacar_feature(self ,  features , targets , ids  ):

        for feature , target , idd in zip(features,targets, ids):
           
            yield {
                'ids': tf.train.Feature(
                    int64_list = tf.train.Int64List( value = [ idd ] )) ,
                
                'feature' : tf.train.Feature(
                    int64_list = tf.train.Int64List( value =  feature   ) ) ,
                
                'target' : tf.train.Feature(
                    int64_list = tf.train.Int64List( value =  target   ) )
                
            }
        
    def dataset_tofile(self , features , filename ):

        writer = tf.python_io.TFRecordWriter(
            filename ,
            options = tf.python_io.TFRecordOptions(
                compression_type = TFRecordCompressionType.GZIP 
            )
        )
        with writer:
            for feature in features:
                
                writer.write(  tf.train.Example(features = tf.train.Features(
                    feature = feature
                ) ).SerializeToString()  )

                
        
    def fill_data_prior( self ):

        self.df_prior = pd.read_csv( self.prior_path , names=['order_id' , 'product_id' ]    )
        
        
        
    def read_products(self ):

        self.df_products = pd.read_csv( self.product_path , names=['product_id' ] )
        self.num_products = self.df_products.shape[0] # number of products
        print( self.num_products )


    def product_to_onehot(  self , product_id ):

        inds = np.where( self.df_products['products_id'] == product_id )
        one_hot = tf.one_hot( inds , self.num_products , dtype = tf.float32 )

        
        return one_hot
    

    def get_prior_products( self , order_id ):

        # find order_ids in prior data set.
        products = self.df_prior.loc[ order_id ]
        ll = products.tolist()
        
        



dataparser = DataParser( "../data")
