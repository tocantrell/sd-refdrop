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

current_extension_directory = scripts.basedir()

class Script(scripts.Script):  

    def title(self):

        return "RefDrop"

    def show(self, is_txt2img):

        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Accordion("RefDrop", open=False):
                enabled = gr.Checkbox(label="Enabled", value=False)
                save_or_use = gr.Radio(["Save", "Use"],label="Mode",
                    info="You must first generate a single image to record its embedding information. Caution: Running \"Save\" a second time will overwrite existing data.")
                rfg = gr.Slider(minimum=-1.0, maximum=1.0, step=0.01, value=0.,
                label="RFG Coefficent", info="RFG is only used applying to a new image. Positive values increase consistency with the saved data while negative do the opposite.")
                save_percent = gr.Slider(minimum=0, maximum=100, step=1, value=100,
                label="Save Percentage", info="Reduce run time by limiting the number of embedding files saved. Minimal impact >=50%")
                

        return [enabled, rfg, save_or_use, save_percent]

    def before_process_batch(self, p, enabled, rfg, save_or_use, save_percent, *args, **kwargs):
    
        if enabled:
            print('RefDrop Enabled')
            CrossAttention.v_count = 0
            CrossAttention.k_count = 0
            
            if save_or_use == 'Use':
                print('Applying RefDrop data')
                CrossAttention.refdrop = 'Use'
                CrossAttention.rfg = rfg
                
            if save_or_use == 'Save':
                print('Saving RefDrop data')
                CrossAttention.refdrop = 'Save'
                
                files = glob.glob(current_extension_directory+'/latents/k/*.pt')
                for f in files:
                    os.remove(f)
                files = glob.glob(current_extension_directory+'/latents/v/*.pt')
                for f in files:
                    os.remove(f)
            
            
            def _forwardBasicTransformerBlock(self, x, context=None, transformer_options={}):
                extra_options = {}
                block = transformer_options.get("block", None)
                block_index = transformer_options.get("block_index", 0)
                transformer_patches = {}
                transformer_patches_replace = {}


                if CrossAttention.refdrop in ['Save','Use']:
                    k_file = current_extension_directory+'/latents/k/'+str(CrossAttention.k_count)+'.pt'
                    v_file = current_extension_directory+'/latents/v/'+str(CrossAttention.v_count)+'.pt'

                refdrop_save = False
                refdrop_use = False
                if CrossAttention.refdrop == 'Use':
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
                elif CrossAttention.refdrop == 'Save':
                    v_refdrop = None
                    k_refdrop = None
                    refdrop_save = True
                else:
                    v_refdrop = None
                    k_refdrop = None


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
                    n = self.attn1.to_out(n,k_refdrop=k_refdrop,v_refdrop=v_refdrop,refdrop_save=refdrop_save,refdrop_use=refdrop_use)
                else:
                    n = self.attn1(n, context=context_attn1, value=value_attn1, transformer_options=extra_options,k_refdrop=k_refdrop,v_refdrop=v_refdrop,refdrop_save=refdrop_save,refdrop_use=refdrop_use)

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

                if CrossAttention.refdrop in ['Save','Use']:
                    CrossAttention.v_count += 1
                    CrossAttention.k_count += 1

                return x
            
            BasicTransformerBlock._forward = _forwardBasicTransformerBlock
         
                    
            def forward_crossattention(self, x, context=None, value=None, mask=None, transformer_options=None,k_refdrop=None,v_refdrop=None,refdrop_save=False,refdrop_use=False):

                if CrossAttention.refdrop in ['Save','Use']:
                    k_file = current_extension_directory+'/latents/k/'+str(CrossAttention.k_count)+'.pt'
                    v_file = current_extension_directory+'/latents/v/'+str(CrossAttention.v_count)+'.pt'

                q = self.to_q(x)
                context = attention.default(context, x)
                k = self.to_k(context)
                if value is not None:
                    v = self.to_v(value)
                    del value
                else:
                    v = self.to_v(context)

                if refdrop_save:
                    #Save K and V to files
                    torch.save(k, k_file)
                    torch.save(v, v_file)

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

            CrossAttention.v_count = 0
            CrossAttention.k_count = 0
            CrossAttention.refdrop = None
            CrossAttention.rfg = rfg
       
    def postprocess_image(self, p, pp, enabled, rfg, save_or_use, save_percent, *args):
        
        if enabled:
            if save_or_use == 'Save':
                if save_percent != 100:
                    save_percent /= 100
                    
                    files = glob.glob(current_extension_directory+'/latents/k/*.pt')
                    l = int(len(files) * save_percent)
                    for f in range(l,len(files)):
                        os.remove(current_extension_directory+'/latents/k/'+str(f)+'.pt')
                        os.remove(current_extension_directory+'/latents/v/'+str(f)+'.pt')
