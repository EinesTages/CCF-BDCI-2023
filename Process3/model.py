import pgl
import math
import paddle.fluid.layers as L
import pgl.layers.conv as conv
import paddle.fluid as F
from transformer_gat_pgl import transformer_gat_pgl

def get_norm(indegree):
    float_degree = L.cast(indegree, dtype="float32")
    float_degree = L.clamp(float_degree, min=1.0)
    norm = L.pow(float_degree, factor=-0.5) 
    return norm
    

class GCN(object):
    """Implement of GCN
    """
    def __init__(self, config, num_class):
        self.num_class = num_class
        self.num_layers = config.get("num_layers", 1)
        self.hidden_size = config.get("hidden_size", 64)
        self.dropout = config.get("dropout", 0.5)
        self.edge_dropout = config.get("edge_dropout", 0.0)

    def forward(self, graph_wrapper, feature, phase):
        
        for i in range(self.num_layers):

            if phase == "train":
                ngw = pgl.sample.edge_drop(graph_wrapper, self.edge_dropout) 
                norm = get_norm(ngw.indegree())
            else:
                ngw = graph_wrapper
                norm = graph_wrapper.node_feat["norm"]


            feature = pgl.layers.gcn(ngw,
                feature,
                self.hidden_size,
                activation="relu",
                norm=norm,
                name="layer_%s" % i)

            feature = L.dropout(
                    feature,
                    self.dropout,
                    dropout_implementation='upscale_in_train')

        if phase == "train": 
            ngw = pgl.sample.edge_drop(graph_wrapper, self.edge_dropout) 
            norm = get_norm(ngw.indegree())
        else:
            ngw = graph_wrapper
            norm = graph_wrapper.node_feat["norm"]

        feature = conv.gcn(ngw,
                     feature,
                     self.num_class,
                     activation=None,
                     norm=norm,
                     name="output")

        return feature

class GAT(object):
    """Implement of GAT"""
    def __init__(self, config, num_class):
        self.num_class = num_class 
        self.num_layers = config.get("num_layers", 1)
        self.num_heads = config.get("num_heads", 8)
        self.hidden_size = config.get("hidden_size", 8)
        self.feat_dropout = config.get("feat_drop", 0.6)
        self.attn_dropout = config.get("attn_drop", 0.6)
        self.edge_dropout = config.get("edge_dropout", 0.0)

    def forward(self, graph_wrapper, feature, phase):
        if phase == "train": 
            edge_dropout = self.edge_dropout
        else:
            edge_dropout = 0

        for i in range(self.num_layers):
            ngw = pgl.sample.edge_drop(graph_wrapper, edge_dropout) 
                
            feature = conv.gat(ngw,
                                feature,
                                self.hidden_size,
                                activation="elu",
                                name="gat_layer_%s" % i,
                                num_heads=self.num_heads,
                                feat_drop=self.feat_dropout,
                                attn_drop=self.attn_dropout)

        ngw = pgl.sample.edge_drop(graph_wrapper, edge_dropout) 
        feature = conv.gat(ngw,
                     feature,
                     self.num_class,
                     num_heads=1,
                     activation=None,
                     feat_drop=self.feat_dropout,
                     attn_drop=self.attn_dropout,
                     name="output")
        return feature

   
class APPNP(object):
    """Implement of APPNP"""
    def __init__(self, config, num_class):
        self.num_class = num_class
        self.num_layers = config.get("num_layers", 1)
        self.hidden_size = config.get("hidden_size", 64)
        self.dropout = config.get("dropout", 0.5)
        self.alpha = config.get("alpha", 0.1)
        self.k_hop = config.get("k_hop", 10)
        self.edge_dropout = config.get("edge_dropout", 0.0)

    def forward(self, graph_wrapper, feature, phase):
        if phase == "train": 
            edge_dropout = self.edge_dropout
        else:
            edge_dropout = 0

        for i in range(self.num_layers):
            feature = L.dropout(
                feature,
                self.dropout,
                dropout_implementation='upscale_in_train')
            feature = L.fc(feature, self.hidden_size, act="relu", name="lin%s" % i)

        feature = L.dropout(
            feature,
            self.dropout,
            dropout_implementation='upscale_in_train')

        feature = L.fc(feature, self.num_class, act=None, name="output")

        feature = conv.appnp(graph_wrapper,
            feature=feature,
            edge_dropout=edge_dropout,
            alpha=self.alpha,
            k_hop=self.k_hop)
        return feature

class SGC(object):
    """Implement of SGC"""
    def __init__(self, config, num_class):
        self.num_class = num_class
        self.num_layers = config.get("num_layers", 1)

    def forward(self, graph_wrapper, feature, phase):
        feature = conv.appnp(graph_wrapper,
            feature=feature,
            edge_dropout=0,
            alpha=0,
            k_hop=self.num_layers)
        feature.stop_gradient=True
        feature = L.fc(feature, self.num_class, act=None, bias_attr=False, name="output")
        return feature

 
class GCNII(object):
    """Implement of GCNII"""
    def __init__(self, config, num_class):
        self.num_class = num_class
        self.num_layers = config.get("num_layers", 1)
        self.hidden_size = config.get("hidden_size", 64)
        self.dropout = config.get("dropout", 0.6)
        self.alpha = config.get("alpha", 0.1)
        self.lambda_l = config.get("lambda_l", 0.5)
        self.k_hop = config.get("k_hop", 64)
        self.edge_dropout = config.get("edge_dropout", 0.0)

    def forward(self, graph_wrapper, feature, phase):
        if phase == "train": 
            edge_dropout = self.edge_dropout
        else:
            edge_dropout = 0

        for i in range(self.num_layers):
            feature = L.fc(feature, self.hidden_size, act="relu", name="lin%s" % i)
            feature = L.dropout(
                feature,
                self.dropout,
                dropout_implementation='upscale_in_train')

        feature = conv.gcnii(graph_wrapper,
            feature=feature,
            name="gcnii",
            activation="relu",
            lambda_l=self.lambda_l,
            alpha=self.alpha,
            dropout=self.dropout,
            k_hop=self.k_hop)


class ResGAT(object):
    """Implement of ResGAT"""
    def __init__(self, config, num_class):
        self.num_class = num_class 
        self.num_layers = config.get("num_layers", 1)
        self.num_heads = config.get("num_heads", 8)
        self.hidden_size = config.get("hidden_size", 8)
        self.feat_dropout = config.get("feat_drop", 0.6)
        self.attn_dropout = config.get("attn_drop", 0.6)
        self.edge_dropout = config.get("edge_dropout", 0.0)

    def forward(self, graph_wrapper, feature, phase):
        # feature [num_nodes, 100]
        if phase == "train": 
            edge_dropout = self.edge_dropout
        else:
            edge_dropout = 0
        feature = L.fc(feature, size=self.hidden_size * self.num_heads, name="init_feature")
        for i in range(self.num_layers):
            ngw = pgl.sample.edge_drop(graph_wrapper, edge_dropout) 
            
            res_feature = feature
            # res_feature [num_nodes, hidden_size * n_heads]
            feature = conv.gat(ngw,
                                feature,
                                self.hidden_size,
                                activation=None,
                                name="gat_layer_%s" % i,
                                num_heads=self.num_heads,
                                feat_drop=self.feat_dropout,
                                attn_drop=self.attn_dropout)
            # feature [num_nodes, num_heads * hidden_size]
            feature = res_feature + feature 
            # [num_nodes, num_heads * hidden_size] + [ num_nodes, hidden_size * n_heads]
            feature = L.relu(feature)
            feature = L.layer_norm(feature, name="ln_%s" % i)

        ngw = pgl.sample.edge_drop(graph_wrapper, edge_dropout) 
        feature = conv.gat(ngw,
                     feature,
                     self.num_class,
                     num_heads=1,
                     activation=None,
                     feat_drop=self.feat_dropout,
                     attn_drop=self.attn_dropout,
                     name="output")
        return feature


class ResGCN(object):
    """Implement of GCN
    """
    def __init__(self, config, num_class):
        self.num_class = num_class
        self.num_layers = config.get("num_layers", 1)
        self.hidden_size = config.get("hidden_size", 64)
        self.dropout = config.get("dropout", 0.5)
        self.edge_dropout = config.get("edge_dropout", 0.0)

    def forward(self, graph_wrapper, feature, phase):
        
        for i in range(self.num_layers):
            
            if phase == "train":
                ngw = pgl.sample.edge_drop(graph_wrapper, self.edge_dropout) 
                norm = get_norm(ngw.indegree())
            else:
                ngw = graph_wrapper
                norm = graph_wrapper.node_feat["norm"]

            res_feature = L.fc(feature, size=self.hidden_size, name="res_feature")
            
            feature = pgl.layers.gcn(ngw,
                feature,
                self.hidden_size,
                activation="relu",
                norm=norm,
                name="layer_%s" % i)

            feature = res_feature + feature 

            feature = L.dropout(
                    feature,
                    self.dropout,
                    dropout_implementation='upscale_in_train')

        if phase == "train": 
            ngw = pgl.sample.edge_drop(graph_wrapper, self.edge_dropout) 
            norm = get_norm(ngw.indegree())
        else:
            ngw = graph_wrapper
            norm = graph_wrapper.node_feat["norm"]

        feature = conv.gcn(ngw,
                     feature,
                     self.num_class,
                     activation=None,
                     norm=norm,
                     name="output")

        return feature

        feature = L.fc(feature, self.num_class, act=None, name="output")
        return feature


class UniMP(object):
    def __init__(self, config, num_class):
        self.num_class = num_class
        self.num_layers = config.get("num_layers", 2)
        self.hidden_size = config.get("hidden_size", 64)
        self.out_size=config.get("out_size", 40)
        self.embed_size=config.get("embed_size", 768)
        self.heads = config.get("heads", 8) 
        self.dropout = config.get("dropout", 0.3)
        self.edge_dropout = config.get("edge_dropout", 0.0)
        self.use_label_e = config.get("use_label_e", False)
            
    
    def embed_input(self, feature):
        
        lay_norm_attr=F.ParamAttr(initializer=F.initializer.ConstantInitializer(value=1))
        lay_norm_bias=F.ParamAttr(initializer=F.initializer.ConstantInitializer(value=0))
        feature=L.layer_norm(feature, name='layer_norm_feature_input', 
                                      param_attr=lay_norm_attr, 
                                      bias_attr=lay_norm_bias)
        
        return feature

    def label_embed_input(self, feature):
        label = F.data(name="label", shape=[None, 1], dtype="int64")
        label_idx = F.data(name='label_idx', shape=[None, 1], dtype="int64")

        label = L.reshape(label, shape=[-1])
        label_idx = L.reshape(label_idx, shape=[-1])

        embed_attr = F.ParamAttr(initializer=F.initializer.NormalInitializer(loc=0.0, scale=1.0))
        embed = F.embedding(input=label, size=(self.out_size, self.embed_size), param_attr=embed_attr )

        feature_label = L.gather(feature, label_idx, overwrite=False)
        feature_label = feature_label + embed
        feature = L.scatter(feature, label_idx, feature_label, overwrite=True)

        
        lay_norm_attr = F.ParamAttr(initializer=F.initializer.ConstantInitializer(value=1))
        lay_norm_bias = F.ParamAttr(initializer=F.initializer.ConstantInitializer(value=0))
        feature = L.layer_norm(feature, name='layer_norm_feature_input', 
                                      param_attr=lay_norm_attr, 
                                      bias_attr=lay_norm_bias)
        return feature
    
    def get_gat_layer(self, i, gw, feature, hidden_size, num_heads, concat=True,
                      layer_norm=True, relu=True, gate=False):
        fan_in=feature.shape[-1]
        bias_bound = 1.0 / math.sqrt(fan_in)
        fc_bias_attr = F.ParamAttr(initializer=F.initializer.UniformInitializer(low=-bias_bound, high=bias_bound))
        
        negative_slope = math.sqrt(5)
        gain = math.sqrt(2.0 / (1 + negative_slope ** 2))
        std = gain / math.sqrt(fan_in)
        weight_bound = math.sqrt(3.0) * std
        fc_w_attr = F.ParamAttr(initializer=F.initializer.UniformInitializer(low=-weight_bound, high=weight_bound))
        
        if concat:
            skip_feature = L.fc(feature,
                         hidden_size * num_heads,
                           param_attr=fc_w_attr,
                           name='fc_skip_' + str(i),
                           bias_attr=fc_bias_attr)
        else:
            skip_feature = L.fc(feature,
                         hidden_size,
                           param_attr=fc_w_attr,
                           name='fc_skip_' + str(i),
                           bias_attr=fc_bias_attr)
        out_feat = transformer_gat_pgl(gw, feature, hidden_size, 'gat_' + str(i), num_heads, concat=concat) 
        # out_feat= out_feat + skip_feature
        
        if gate: 
            fan_in = out_feat.shape[-1]*3
            bias_bound = 1.0 / math.sqrt(fan_in)
            fc_bias_attr = F.ParamAttr(initializer=F.initializer.UniformInitializer(low=-bias_bound, high=bias_bound))

            negative_slope = math.sqrt(5)
            gain = math.sqrt(2.0 / (1 + negative_slope ** 2))
            std = gain / math.sqrt(fan_in)
            weight_bound = math.sqrt(3.0) * std
            fc_w_attr = F.ParamAttr(initializer=F.initializer.UniformInitializer(low=-weight_bound, high=weight_bound))
            gate_f = L.fc([skip_feature, out_feat, out_feat - skip_feature], 1,
                           param_attr=fc_w_attr,
                           name='gate_' + str(i),
                           bias_attr=fc_bias_attr)
            
            gate_f = L.sigmoid(gate_f) 
            out_feat = skip_feature * gate_f + out_feat * (1 - gate_f)
        else:
            out_feat = out_feat + skip_feature
                 
        if layer_norm:
            lay_norm_attr = F.ParamAttr(initializer=F.initializer.ConstantInitializer(value=1))
            lay_norm_bias = F.ParamAttr(initializer=F.initializer.ConstantInitializer(value=0))
            out_feat = L.layer_norm(out_feat, name='layer_norm_' + str(i), 
                                      param_attr=lay_norm_attr, 
                                      bias_attr=lay_norm_bias)
        if relu:
            out_feat = L.relu(out_feat)
        return out_feat
        
    def forward(self, graph_wrapper, feature, phase):
        if phase == "train": 
            edge_dropout = self.edge_dropout
            dropout = self.dropout
        else:
            edge_dropout = 0
            dropout = 0

        if self.use_label_e:
            feature = self.label_embed_input(feature)
            gate = True
        else:
            feature = self.embed_input(feature)
            gate = False
        if dropout > 0:
            feature = L.dropout(feature, dropout_prob=dropout, 
                                    dropout_implementation='upscale_in_train')
        for i in range(self.num_layers - 1):
            ngw = pgl.sample.edge_drop(graph_wrapper, edge_dropout) 
            feature = self.get_gat_layer(i, ngw, feature, 
                                             hidden_size=self.hidden_size,
                                             num_heads=self.heads, 
                                             concat=True, 
                                             layer_norm=True, relu=True, gate=gate)
            if dropout > 0:
                feature = L.dropout(feature, dropout_prob=self.dropout, 
                                     dropout_implementation='upscale_in_train') 

        feature = self.get_gat_layer(self.num_layers - 1, ngw, feature, 
                                           hidden_size=self.out_size, num_heads=self.heads, 
                                             concat=False, layer_norm=False, relu=False, gate=True)
  
        pred = L.fc(
            feature, self.num_class, act=None, name="pred_output")
        return pred