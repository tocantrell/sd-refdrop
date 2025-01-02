import os
from pathlib import Path

import numpy as np

import modules.scripts as scripts
from ldm_patched.ldm.modules.attention import CrossAttention
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

    #def show(self, is_txt2img):

    #    return is_txt2img

    def ui(self, is_img2img):
        
        enabled = gr.Checkbox(label="Enabled", value=False)
        rfg = gr.Slider(minimum=-1.0, maximum=1.0, step=0.01, value=0.,
        label="RFG Coefficent", info="RFG is only used applying to a new image. Positive values increase consistency with the saved data while negative do the opposite.")
        save_or_use = gr.Radio(["Save", "Use"],label="Mode",
            info="You must first generate a single image to record its latent information. Caution: Running \"Save\" a second time will overwrite existing data.")

        return [enabled, rfg, save_or_use]

    def run(self, p, enabled, rfg, save_or_use):

        if enabled == False:

            CrossAttention.v_count = 0
            CrossAttention.k_count = 0
            CrossAttention.refdrop = None
            CrossAttention.rfg = rfg

        else:

            k_folder = 'latents/k/'
            v_folder = 'latents/v/'

            CrossAttention.v_count = 0
            CrossAttention.k_count = 0
            CrossAttention.refdrop = None
            CrossAttention.rfg = rfg

            #Redefine the crossattention forward method to now save or load previous K and V tensors during run
            def forward(self, x, context=None, value=None, mask=None, transformer_options=None):

                if CrossAttention.refdrop in ['Save','Use']:
                    k_file = current_extension_directory+'/'+k_folder+str(CrossAttention.k_count)+'.pt'
                    v_file = current_extension_directory+'/'+v_folder+str(CrossAttention.v_count)+'.pt'

                    CrossAttention.v_count += 1
                    CrossAttention.k_count += 1

                q = self.to_q(x)
                context = attention.default(context, x)
                k = self.to_k(context)
                if value is not None:
                    v = self.to_v(value)
                    del value
                else:
                    v = self.to_v(context)

                if CrossAttention.refdrop == 'Save':
                    #Save K and V to files
                    torch.save(k, k_file)
                    torch.save(v, v_file)
                elif CrossAttention.refdrop == 'Use':
                    v_prev = torch.load(v_file, weights_only=True)
                    k_prev = torch.load(k_file, weights_only=True)

                if mask is None:
                    out = attention.optimized_attention(q, k, v, self.heads)
                    if CrossAttention.refdrop == 'Use':
                        out_prev = attention.optimized_attention(q, k_prev, v_prev, self.heads)
                else:
                    out = attention.optimized_attention_masked(q, k, v, self.heads, mask)
                    if CrossAttention.refdrop == 'Use':
                        out_prev = attention.optimized_attention(q, k_prev, v_prev, self.heads)
                    
                if CrossAttention.refdrop == 'Use':
                    out = (out * (1-CrossAttention.rfg)) + (out_prev * CrossAttention.rfg)

                return self.to_out(out)

            CrossAttention.forward = forward

            if save_or_use == 'Save':

                #First delete potential existing latent data
                files = glob.glob(current_extension_directory+'/'+k_folder+'*.pt')
                print(current_extension_directory+'/'+k_folder)
                print(files)
                for f in files:
                    print(f)
                    os.remove(f)
                files = glob.glob(current_extension_directory+'/'+v_folder+'*.pt')
                for f in files:
                    os.remove(f)

                CrossAttention.v_count = 0
                CrossAttention.k_count = 0
                CrossAttention.refdrop = 'Save'

            if save_or_use == 'Use':

                CrossAttention.v_count = 0
                CrossAttention.k_count = 0
                CrossAttention.refdrop = 'Use'

                #Check for saved latent data if in use state
                run_ok = False
                #if save_or_use == 'Use':
                #    if len(os.listdir(k_folder))==0:
                #        run_ok = True
                #    if len(os.listdir(v_folder))==0:
                #        run_ok = True

                if run_ok:
                    raise Exception('No image latent data has been saved. First run RefDrop in "Save" mode for a single seed.')

        proc = process_images(p)

        return proc
