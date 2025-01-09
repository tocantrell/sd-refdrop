import os
from pathlib import Path

import numpy as np

import modules.scripts as scripts
from ldm_patched.ldm.modules.attention import CrossAttention, BasicTransformerBlock
import ldm_patched.ldm.modules.attention as attention
import gradio as gr

import modules.scripts as scripts
from modules.processing import process_images, Processed

import torch
import glob

import ldm_patched.modules.ops
ops = ldm_patched.modules.ops.disable_weight_init

current_extension_directory = scripts.basedir()

def remove_latent_files(save_loc):
    if save_loc == 'Disk':
        files = glob.glob(current_extension_directory+'/latents/k/*.pt')
        for f in files:
            os.remove(f)
        files = glob.glob(current_extension_directory+'/latents/v/*.pt')
        for f in files:
            os.remove(f)
    else:
        try:
            del CrossAttention.k_dict
        except:
            pass
        try:
            del CrossAttention.v_dict
        except:
            pass
        CrossAttention.k_dict = {}
        CrossAttention.v_dict = {}

def remove_all_latents():
    remove_latent_files('Disk')
    remove_latent_files('RAM')

class Script(scripts.Script):  

    def title(self):

        return "RefDrop"

    def show(self, is_txt2img):

        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Accordion("RefDrop", open=False):
                enabled = gr.Checkbox(label="Enabled", value=False)
                
                with gr.Row(equal_height=True):
                    save_or_use = gr.Radio(["Save", "Use"],value="Save",label="Mode",
                                        info="You must first generate a single image to record its embedding information. Caution: Running \"Save\" a second time will overwrite existing data.")
                    enabled_hr = gr.Checkbox(label="Enabled for hires fix", value=False)
                    
                rfg = gr.Slider(minimum=-1.0, maximum=1.0, step=0.01, value=0.,
                                label="RFG Coefficent",
                                info="RFG is only used applying to a new image. Positive values increase consistency with the saved data while negative do the opposite.")
                
                with gr.Row(equal_height=True):
                    save_loc = gr.Radio(["RAM", "Disk"],value="RAM",label="Latent Store Location",info="Choose 'Disk' if low on memory.")
                    save_percent = gr.Slider(minimum=0, maximum=100, step=1, value=100,
                                            label="Save Percentage",
                                            info="Reduce run time by limiting the number of embedding files saved. Minimal impact >=50%")
                with gr.Row():                         
                    layer_input = gr.Checkbox(label='input',value=True,info='Select which layer group to use. Use mode must use layers that were selected during Save mode, though fewer may be selected during Use mode.')
                    delete_button = gr.Button('Delete Saved RefDrop Latents',size='sm', scale=0)
                layer_middle = gr.Checkbox(label='middle',value=True)
                layer_output = gr.Checkbox(label='output',value=True)
                
                
        delete_button.click(remove_all_latents)
        
        return [enabled, rfg, save_or_use, enabled_hr, save_loc, save_percent, layer_input,layer_middle,layer_output]

    def process_before_every_step(
            self,
            p,
            enabled,
            rfg,
            save_or_use,
            enabled_hr,
            save_loc,
            save_percent,
            layer_input,
            layer_middle,
            layer_output,
            *args,
            **kwarg
        ):

        if enabled:
            CrossAttention.current_step += 1
            CrossAttention.layer_index = 0
            
       
    def process_before_every_sampling(
                self,
                p,
                enabled,
                rfg,
                save_or_use,
                enabled_hr,
                save_loc,
                save_percent,
                layer_input,
                layer_middle,
                layer_output,
                *args,
                **kwarg
            ):
        
        if enabled:
            #Reset the layer name to the start
            CrossAttention.layer_name = 'input'
            CrossAttention.layer_index = 0
            
            #Disable after initial run if "Enable for hires" not selected
            if p.is_hr_pass:
                if not enabled_hr:
                    print('Not using RefDrop for hires fix')
                    CrossAttention.refdrop = 'Done'
                else:
                    CrossAttention.refdrop_hires = True
        


    def before_process_batch(
            self,
            p,
            enabled,
            rfg,
            save_or_use,
            enabled_hr,
            save_loc,
            save_percent,
            layer_input,
            layer_middle,
            layer_output,
            *args,
            **kwarg
        ):
        
        
        layer_list = ['input','middle','output']
        CrossAttention.layer_refdrop = [x for x in layer_list if [layer_input,layer_middle,layer_output][layer_list.index(x)]]
        
        if enabled:
            print('RefDrop Enabled')
            
            CrossAttention.current_step = 0
            if save_percent != 100:
                save_percent /= 100.
                CrossAttention.max_step = int(p.steps * save_percent)
            else:
                CrossAttention.max_step = p.steps
            
            CrossAttention.rfg = rfg
            CrossAttention.current_step = 0
            CrossAttention.layer_name = 'input'
            CrossAttention.layer_index = 0
            CrossAttention.refdrop_hires = False
            CrossAttention.to_disk = False
            if save_loc == 'Disk':
                CrossAttention.to_disk = True
            
            if save_or_use == 'Use':
                print('Applying RefDrop data')
                CrossAttention.refdrop = 'Use'
                
            if save_or_use == 'Save':
                print('Saving RefDrop data')
                CrossAttention.refdrop = 'Save'
                #Delete existing latent data
                remove_latent_files(save_loc)
            
            
            def _forwardBasicTransformerBlock(self, x, context=None, transformer_options={}):
                extra_options = {}
                block = transformer_options.get("block", None)
                block_index = transformer_options.get("block_index", 0)
                transformer_patches = {}
                transformer_patches_replace = {}
                #print(block)
                
                if CrossAttention.layer_name != block[0]:
                    CrossAttention.layer_name = block[0]
                    CrossAttention.layer_index = 0

                #Define file save or read location
                if CrossAttention.refdrop_hires:
                    hires = '_hires'
                else:
                    hires = ''
                latentname = CrossAttention.layer_name +'_step'+ str(CrossAttention.current_step) +'_layer'+ str(CrossAttention.layer_index) + hires
                if CrossAttention.to_disk:
                    k_file = current_extension_directory + '/latents/k/' + latentname + '.pt'
                    v_file = current_extension_directory + '/latents/v/' + latentname + '.pt'
                else:
                    k_file = latentname
                    v_file = latentname
                refdrop_save = False
                refdrop_use = False
                
                if CrossAttention.max_step <= CrossAttention.current_step:
                    CrossAttention.refdrop = 'Done'
                    v_refdrop = None
                    k_refdrop = None
                
                if (CrossAttention.refdrop == 'Use')&(CrossAttention.to_disk)&(CrossAttention.layer_name in CrossAttention.layer_refdrop):
                    try:
                        v_refdrop = torch.load(v_file, weights_only=True)
                        k_refdrop = torch.load(k_file, weights_only=True)
                        refdrop_use = True
                    except:
                        #Running without the last few K and V files will not significantly change the results.
                        #Also allows for variable hires fix and adetailer
                        print('Saved RefDrop file not found. Continuing without RefDrop.')
                        CrossAttention.refdrop = 'Done'
                        v_refdrop = None
                        k_refdrop = None
                if (CrossAttention.refdrop == 'Use')&(CrossAttention.to_disk!=True)&(CrossAttention.layer_name in CrossAttention.layer_refdrop):
                    try:
                        v_refdrop = CrossAttention.v_dict[v_file].to('cuda')
                        k_refdrop = CrossAttention.k_dict[k_file].to('cuda')
                        refdrop_use = True
                    except:
                        #Running without the last few K and V files will not significantly change the results.
                        #Also allows for variable hires fix and adetailer
                        print('Saved RefDrop file not found. Continuing without RefDrop.')
                        CrossAttention.refdrop = 'Done'
                        v_refdrop = None
                        k_refdrop = None
                elif CrossAttention.refdrop == 'Save':
                    v_refdrop = None
                    k_refdrop = None
                    refdrop_save = True
                else:
                    v_refdrop = None
                    k_refdrop = None
                    refdrop_use = False
                    refdrop_save = False

                for k in transformer_options:
                    if k == "patches":
                        transformer_patches = transformer_options[k]
                    elif k == "patches_replace":
                        transformer_patches_replace = transformer_options[k]
                    else:
                        extra_options[k] = transformer_options[k]

                extra_options["n_heads"] = self.n_heads
                extra_options["dim_head"] = self.d_head

                if self.ff_in:
                    x_skip = x
                    x = self.ff_in(self.norm_in(x))
                    if self.is_res:
                        x += x_skip

                n = self.norm1(x)
                if self.disable_self_attn:
                    context_attn1 = context
                else:
                    context_attn1 = None
                value_attn1 = None

                if "attn1_patch" in transformer_patches:
                    patch = transformer_patches["attn1_patch"]
                    if context_attn1 is None:
                        context_attn1 = n
                    value_attn1 = context_attn1
                    for p in patch:
                        n, context_attn1, value_attn1 = p(n, context_attn1, value_attn1, extra_options)

                if block is not None:
                    transformer_block = (block[0], block[1], block_index)
                else:
                    transformer_block = None
                attn1_replace_patch = transformer_patches_replace.get("attn1", {})
                block_attn1 = transformer_block
                if block_attn1 not in attn1_replace_patch:
                    block_attn1 = block

                if block_attn1 in attn1_replace_patch:
                    if context_attn1 is None:
                        context_attn1 = n
                        value_attn1 = n
                    n = self.attn1.to_q(n)
                    context_attn1 = self.attn1.to_k(context_attn1)
                    value_attn1 = self.attn1.to_v(value_attn1)
                    n = attn1_replace_patch[block_attn1](n, context_attn1, value_attn1, extra_options)
                    n = self.attn1.to_out(n)

                else:
                    #Apply RefDrop if the current layer is in the selected list
                    if CrossAttention.layer_name in CrossAttention.layer_refdrop:
                        n = self.attn1(
                                n,
                                context=context_attn1,
                                value=value_attn1,
                                transformer_options=extra_options,
                                k_refdrop=k_refdrop,
                                v_refdrop=v_refdrop,
                                refdrop_save=refdrop_save,
                                refdrop_use=refdrop_use,
                                k_file=k_file,
                                v_file=v_file
                            )
                    else:
                        n = self.attn1(
                                n,
                                context=context_attn1,
                                value=value_attn1,
                                transformer_options=extra_options
                            )

                if "attn1_output_patch" in transformer_patches:
                    patch = transformer_patches["attn1_output_patch"]
                    for p in patch:
                        n = p(n, extra_options)

                x += n
                if "middle_patch" in transformer_patches:
                    patch = transformer_patches["middle_patch"]
                    for p in patch:
                        x = p(x, extra_options)

                if self.attn2 is not None:
                    n = self.norm2(x)
                    if self.switch_temporal_ca_to_sa:
                        context_attn2 = n
                    else:
                        context_attn2 = context
                    value_attn2 = None
                    if "attn2_patch" in transformer_patches:
                        patch = transformer_patches["attn2_patch"]
                        value_attn2 = context_attn2
                        for p in patch:
                            n, context_attn2, value_attn2 = p(n, context_attn2, value_attn2, extra_options)

                    attn2_replace_patch = transformer_patches_replace.get("attn2", {})
                    block_attn2 = transformer_block
                    if block_attn2 not in attn2_replace_patch:
                        block_attn2 = block

                    if block_attn2 in attn2_replace_patch:
                        if value_attn2 is None:
                            value_attn2 = context_attn2
                        n = self.attn2.to_q(n)
                        context_attn2 = self.attn2.to_k(context_attn2)
                        value_attn2 = self.attn2.to_v(value_attn2)
                        n = attn2_replace_patch[block_attn2](n, context_attn2, value_attn2, extra_options)
                        n = self.attn2.to_out(n)
                    else:
                        n = self.attn2(n, context=context_attn2, value=value_attn2, transformer_options=extra_options)

                if "attn2_output_patch" in transformer_patches:
                    patch = transformer_patches["attn2_output_patch"]
                    for p in patch:
                        n = p(n, extra_options)

                x += n
                if self.is_res:
                    x_skip = x
                x = self.ff(self.norm3(x))
                if self.is_res:
                    x += x_skip

                CrossAttention.layer_index += 1
                
                return x
            
            BasicTransformerBlock._forward = _forwardBasicTransformerBlock
         
                    
            def forward_crossattention(
                    self,
                    x,
                    context=None,
                    value=None,
                    mask=None,
                    transformer_options=None,
                    k_refdrop=None,
                    v_refdrop=None,
                    refdrop_save=False,
                    refdrop_use=False,
                    k_file=None,
                    v_file=None
                ):

                q = self.to_q(x)
                context = attention.default(context, x)
                k = self.to_k(context)
                if value is not None:
                    v = self.to_v(value)
                    del value
                else:
                    v = self.to_v(context)

                if refdrop_save:
                    if CrossAttention.to_disk:
                        #Save K and V to files on disk
                        torch.save(k, k_file)
                        torch.save(v, v_file)
                    else:
                        #Save K and V to files to memory via a dictionary
                        CrossAttention.k_dict.update({k_file:k.to('cpu')})
                        CrossAttention.v_dict.update({v_file:v.to('cpu')})

                if mask is None:
                    out = attention.optimized_attention(q, k, v, self.heads)
                    if refdrop_use:
                        out_refdrop = attention.optimized_attention(q, k_refdrop, v_refdrop, self.heads)
                else:
                    out = attention.optimized_attention_masked(q, k, v, self.heads, mask)
                    if refdrop_use:
                        out_refdrop = attention.optimized_attention(q, k_refdrop, v_refdrop, self.heads)
                    
                if refdrop_use:
                    out = (out * (1-CrossAttention.rfg)) + (out_refdrop * CrossAttention.rfg)
                    
                return self.to_out(out)

            CrossAttention.forward = forward_crossattention
                
        else:

            CrossAttention.layer_index = 0
            CrossAttention.refdrop = None
            CrossAttention.rfg = rfg
            CrossAttention.current_step = 0
            CrossAttention.layer_name = 'input'
  
