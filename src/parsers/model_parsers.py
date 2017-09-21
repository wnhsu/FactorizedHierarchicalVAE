#!/usr/bin/python 

from parser_common import *
from libs.activations import *

nl_dict = {"None": None,
           "tanh": tf.nn.tanh,
           "relu": tf.nn.relu,
           "sigmoid": tf.nn.sigmoid,
           "relu_n10": lambda x: custom_relu(x, cutoff=-10)}
inv_nl_dict = dict(zip(nl_dict.values(), nl_dict.keys()))

def parse_raw_fc_str(raw_fc_str):
    return [int(hu) for hu in raw_fc_str.split(',') if hu]

def fc_conf_to_str(fc_conf):
    return ','.join(map(str, fc_conf))

def parse_raw_conv_str(raw_conv_str):
    """
    raw_conv_str is: 
        #FILTERS,KERNEL_HEIGHT,KERNEL_WIDTH[,STRIDE_HEIGHT,STRIDE_WIDTH[,PADDING]]
    """
    conv = []
    raw_layers = [l.split('_') for l in raw_conv_str.split(',') if l]
    for raw_layer in raw_layers:
        if len(raw_layer) == 3:
            conv.append(tuple([int(x) for x in raw_layer]) + (1, 1, "valid"))
        elif len(raw_layer) == 5:
            conv.append(tuple([int(x) for x in raw_layer]) + ("same",))
        elif len(raw_layer) == 6:
            conv.append(tuple([int(x) for x in raw_layer[:5]]+[raw_layer[5]]))
        else:
            raise ValueError("raw_layer format invalid: %s" % str(raw_layer))
    return conv

def conv_conf_to_str(conv_conf):
    return ",".join(["_".join(map(str, layer)) for layer in conv_conf])

class base_model_parser(base_parser):
    def __init__(self, model_config_path):
        self.parser = DefaultConfigParser()

        parser = self.parser
        config = {}
        if len(parser.read(model_config_path)) == 0:
            raise ValueError("base_model_parser(): %s not found", model_config_path)

        config["input_shape"]   = None
        config["target_shape"]  = None
        config["n_latent"]      = parser.getint("model", "n_latent")
        config["x_conti"]       = parser.getboolean("model", "x_conti")
        config["x_mu_nl"]       = None
        config["x_logvar_nl"]   = None 
        config["n_bins"]        = None
        config["if_bn"]         = parser.getboolean("model", "if_bn", True)

        if config["x_conti"]:
            config["x_mu_nl"] = nl_dict[parser.get("model", "x_mu_nl")]
            config["x_logvar_nl"] = nl_dict[parser.get("model", "x_logvar_nl")]
        else:
            config["n_bins"] = parser.getint("model", "n_bins")
        
        self.config = config
    
    @staticmethod
    def write_config(config, f):
        f.write("[model]\n")
        for key in ["n_latent", "if_bn", "x_conti"]:
            f.write("%s= %s\n" % (key.ljust(20), str(config[key])))
        if config["x_conti"]:
            for key in ["x_mu_nl", "x_logvar_nl"]:
                f.write("%s= %s\n" % (key.ljust(20), str(inv_nl_dict[config[key]])))
        else:
            for key in ["n_bins"]:
                f.write("%s= %s\n" % (key.ljust(20), str(config[key])))
        
class fvae_model_parser(base_model_parser):
    def __init__(self, model_config_path):
        super(fvae_model_parser, self).__init__(model_config_path)

        self.config["hu_enc"]   = parse_raw_fc_str(
                self.parser.get("model", "hu_enc", ""))

    def get_config(self):
        return self.config

    @staticmethod
    def write_config(config, f):
        super(fvae_model_parser, fvae_model_parser).write_config(config, f)
        f.write("\n")
        for key in ["hu_enc"]:
            f.write("%s= %s\n" % (key.ljust(20), fc_conf_to_str(config[key])))

class cvae_model_parser(fvae_model_parser):
    def __init__(self, model_config_path):
        super(cvae_model_parser, self).__init__(model_config_path)

        self.config["conv_enc"] = parse_raw_conv_str(
                self.parser.get("model", "conv_enc", ""))

    @staticmethod
    def write_config(config, f):
        super(cvae_model_parser, cvae_model_parser).write_config(config, f)
        f.write("\n")
        for key in ["conv_enc"]:
            f.write("%s= %s\n" % (key.ljust(20), conv_conf_to_str(config[key])))

class rvae_model_parser(base_model_parser):
    def __init__(self, model_config_path):
        super(rvae_model_parser, self).__init__(model_config_path)

        self.config["hu_enc"] = parse_raw_fc_str(
                self.parser.get("model", "hu_enc", ""))
        self.config["hu_dec"] = parse_raw_fc_str(
                self.parser.get("model", "hu_dec", ""))
        self.config["rec_learn_init"] = self.parser.getboolean(
                "model", "rec_learn_init", False)
        self.config["rec_enc"] = parse_raw_fc_str(
                self.parser.get("model", "rec_enc", ""))
        self.config["rec_enc_concur"] = self.parser.getint(
                "model", "rec_enc_concur", 1)
        self.config["rec_enc_out"] = self.parser.get(
                "model", "rec_enc_out", "all_h")
        self.config["rec_enc_bi"] = self.parser.getboolean(
                "model", "rec_enc_bi", False)
        self.config["rec_dec"] = parse_raw_fc_str(
                self.parser.get("model", "rec_dec", ""))
        self.config["rec_dec_bi"] = self.parser.getboolean(
                "model", "rec_dec_bi", False)
        self.config["rec_dec_inp_train"] = self.parser.get(
                "model", "rec_dec_inp_train", "")
        self.config["rec_dec_inp_test"] = self.parser.get(
                "model", "rec_dec_inp_test", "")
        self.config["rec_cell_type"] = self.parser.get(
                "model", "rec_cell_type", "gru")
        self.config["rec_dec_concur"] = self.parser.getint(
                "model", "rec_dec_concur", 1)
        self.config["rec_dec_inp_hist"] = self.parser.getint(
                "model", "rec_dec_inp_hist", 1)

    def get_config(self):
        return self.config

    @staticmethod
    def write_config(config, f):
        super(rvae_model_parser, rvae_model_parser).write_config(config, f)
        f.write("\n")
        for key in ["rec_learn_init", "rec_cell_type"]:
            f.write("%s= %s\n" % (key.ljust(20), config[key]))

        f.write("\n")
        for key in ["hu_enc", "rec_enc"]:
            f.write("%s= %s\n" % (key.ljust(20), fc_conf_to_str(config[key])))
        for key in ["rec_enc_concur", "rec_enc_out", "rec_enc_bi"]:
            f.write("%s= %s\n" % (key.ljust(20), config[key]))

        f.write("\n")
        for key in ["hu_dec", "rec_dec"]:
            f.write("%s= %s\n" % (key.ljust(20), fc_conf_to_str(config[key])))
        for key in ["rec_dec_bi", "rec_dec_inp_train", "rec_dec_inp_test", 
                "rec_dec_concur", "rec_dec_inp_hist"]:
            f.write("%s= %s\n" % (key.ljust(20), config[key]))

class base_fhvae_model_parser(base_parser):
    def __init__(self, model_config_path):
        self.parser = DefaultConfigParser()

        parser = self.parser
        config = {}
        if len(parser.read(model_config_path)) == 0:
            raise ValueError("base_fhvae_model_parser(): %s not found",
                    model_config_path)

        config["input_shape"]   = None
        config["target_shape"]  = None
        config["n_latent1"]     = parser.getint("model", "n_latent1")
        config["n_latent2"]     = parser.getint("model", "n_latent2")
        config["n_class1"]      = None
        config["latent1_std"]   = parser.getfloat("model", "latent1_std", 0.5)
        config["z1_logvar_nl"]  = nl_dict[parser.get("model", "z1_logvar_nl", "None")]
        config["z2_logvar_nl"]  = nl_dict[parser.get("model", "z2_logvar_nl", "None")]
        config["x_conti"]       = parser.getboolean("model", "x_conti")
        config["x_mu_nl"]       = None
        config["x_logvar_nl"]   = None 
        config["n_bins"]        = None
        config["if_bn"]         = parser.getboolean("model", "if_bn", True)

        if config["x_conti"]:
            config["x_mu_nl"] = nl_dict[parser.get("model", "x_mu_nl")]
            config["x_logvar_nl"] = nl_dict[parser.get("model", "x_logvar_nl")]
        else:
            config["n_bins"] = parser.getint("model", "n_bins")
        
        self.config = config
    
    @staticmethod
    def write_config(config, f):
        f.write("[model]\n")
        for key in ["n_latent1", "n_latent2", "latent1_std", 
                "z1_logvar_nl", "z2_logvar_nl", "if_bn", "x_conti"]:
            f.write("%s= %s\n" % (key.ljust(20), str(config[key])))
        if config["x_conti"]:
            for key in ["x_mu_nl", "x_logvar_nl"]:
                f.write("%s= %s\n" % (key.ljust(20), str(inv_nl_dict[config[key]])))
        else:
            for key in ["n_bins"]:
                f.write("%s= %s\n" % (key.ljust(20), str(config[key])))

class fc_fhvae_model_parser(base_fhvae_model_parser):
    def __init__(self, model_config_path):
        super(fc_fhvae_model_parser, self).__init__(model_config_path)

        self.config["hu_z1_enc"]   = parse_raw_fc_str(
                self.parser.get("model", "hu_z1_enc", ""))
        self.config["hu_z2_enc"]    = parse_raw_fc_str(
                self.parser.get("model", "hu_z2_enc", ""))
        self.config["hu_dec"]       = parse_raw_fc_str(
                self.parser.get("model", "hu_dec", ""))

    def get_config(self):
        return self.config

    @staticmethod
    def write_config(config, f):
        super(fc_fhvae_model_parser, fc_fhvae_model_parser).write_config(config, f)
        f.write("\n")
        for key in ["hu_z1_enc", "hu_z2_enc", "hu_dec"]:
            f.write("%s= %s\n" % (key.ljust(20), fc_conf_to_str(config[key])))

class rec_fhvae_model_parser(base_fhvae_model_parser):
    def __init__(self, model_config_path):
        super(rec_fhvae_model_parser, self).__init__(model_config_path)

        config = self.config
        parser = self.parser

        config["rec_learn_init"]    = parser.getboolean("model", "rec_learn_init", False)
        config["rec_cell_type"]     = parser.get("model", "rec_cell_type", "gru")

        config["rec_z1_enc"]        = parse_raw_fc_str(parser.get("model", "rec_z1_enc", ""))
        config["rec_z1_enc_bi"]     = parser.getboolean("model", "rec_z1_enc_bi", False)
        config["rec_z1_enc_concur"] = parser.getint("model", "rec_z1_enc_concur", 1)
        config["rec_z1_enc_out"]    = parser.get("model", "rec_z1_enc_out", "all_h")
        config["hu_z1_enc"]         = parse_raw_fc_str(parser.get("model", "hu_z1_enc", ""))

        config["rec_z2_enc"]        = parse_raw_fc_str(parser.get("model", "rec_z2_enc", ""))
        config["rec_z2_enc_bi"]     = parser.getboolean("model", "rec_z2_enc_bi", False)
        config["rec_z2_enc_concur"] = parser.getint("model", "rec_z2_enc_concur", 1)
        config["rec_z2_enc_out"]    = parser.get("model", "rec_z2_enc_out", "all_h")
        config["hu_z2_enc"]         = parse_raw_fc_str(parser.get("model", "hu_z2_enc", ""))
        
        config["hu_dec"]            = parse_raw_fc_str(parser.get("model", "hu_dec", ""))
        config["rec_dec"]           = parse_raw_fc_str(parser.get("model", "rec_dec", ""))
        config["rec_dec_bi"]        = parser.getboolean("model", "rec_dec_bi", False)
        config["rec_dec_inp_train"] = parser.get("model", "rec_dec_inp_train", "")
        config["rec_dec_inp_test"]  = parser.get("model", "rec_dec_inp_test", "")
        config["rec_dec_inp_hist"]  = parser.getint("model", "rec_dec_inp_hist", 1)
        config["rec_dec_concur"]    = parser.getint("model", "rec_dec_concur", 1)

        self.config = config

    def get_config(self):
        return self.config

    @staticmethod
    def write_config(config, f):
        super(rec_fhvae_model_parser, rec_fhvae_model_parser).write_config(config, f)
        
        f.write("\n")
        for key in ["rec_learn_init", "rec_cell_type"]:
            f.write("%s= %s\n" % (key.ljust(20), config[key]))

        f.write("\n")
        for key in ["rec_z1_enc", "hu_z1_enc"]:
            f.write("%s= %s\n" % (key.ljust(20), fc_conf_to_str(config[key])))
        for key in ["rec_z1_enc_bi", "rec_z1_enc_concur", "rec_z1_enc_out"]:
            f.write("%s= %s\n" % (key.ljust(20), config[key]))

        f.write("\n")
        for key in ["rec_z2_enc", "hu_z2_enc"]:
            f.write("%s= %s\n" % (key.ljust(20), fc_conf_to_str(config[key])))
        for key in ["rec_z2_enc_bi", "rec_z2_enc_concur", "rec_z2_enc_out"]:
            f.write("%s= %s\n" % (key.ljust(20), config[key]))

        f.write("\n")
        for key in ["hu_dec", "rec_dec"]:
            f.write("%s= %s\n" % (key.ljust(20), fc_conf_to_str(config[key])))
        for key in ["rec_dec_bi", "rec_dec_inp_train", "rec_dec_inp_test",
                "rec_dec_inp_hist", "rec_dec_concur"]:
            f.write("%s= %s\n" % (key.ljust(20), config[key]))
