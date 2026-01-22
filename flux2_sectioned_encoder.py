"""
FLUX.2 Klein Sectioned Text Encoder

Organizes prompts into Front/Mid/End sections with semantic integrity.
Works with Detail Controller's DYNAMIC region calculation (25%/50%/25% of actual tokens).

The encoder just ensures concepts stay together - Detail Controller handles the math.

Features:
- Manual section assignment with [FRONT]/[MID]/[END] markers
- Auto-balanced mode based on sentence distribution
- No fixed token targets - natural length flows through
- Semantic boundaries preserved during tokenization
"""

import torch
import re

try:
    import comfy.sd
    import comfy.utils
    HAS_COMFY = True
except ImportError:
    HAS_COMFY = False


class Flux2KleinSectionedEncoder:
    """
    Encodes prompts with explicit Front/Mid/End section control.
    
    The Detail Controller will automatically apply 25%/50%/25% distribution
    based on the ACTUAL token count from this encoding.
    
    Use section markers:
    [FRONT] subject and main concept
    [MID] detailed descriptions and modifiers  
    [END] style, quality, artistic terms
    
    Or use auto mode to distribute sentences naturally.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "mode": (["manual", "auto_balanced"], {
                    "default": "manual",
                    "tooltip": "Manual: use [FRONT]/[MID]/[END] markers | Auto: distribute sentences by 25/50/25"
                }),
            },
            "optional": {
                "front_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Main subject, primary concept"
                }),
                "mid_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Details, modifiers, attributes"
                }),
                "end_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Style, quality, artistic terms"
                }),
                "combined_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Single prompt with [FRONT]/[MID]/[END] markers, or full text for auto mode"
                }),
                "separator": (["comma", "period", "space", "newline"], {
                    "default": "comma",
                    "tooltip": "How to join sections in final prompt"
                }),
                "show_preview": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Print section breakdown to console"
                }),
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show detailed tokenization info"
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("conditioning", "front_section", "mid_section", "end_section", "full_prompt")
    FUNCTION = "encode_sectioned"
    CATEGORY = "conditioning/flux2klein"
    OUTPUT_NODE = True

    def _estimate_tokens(self, text):
        """Rough token estimation for preview only"""
        if not text or len(text.strip()) == 0:
            return 0
        # Average chars per token (approximate)
        avg_chars = 3.5
        return max(1, int(len(text) / avg_chars))

    def _parse_manual_sections(self, text):
        """Parse [FRONT]/[MID]/[END] markers from combined prompt"""
        pattern = r'\[(FRONT|MID|END)\](.*?)(?=\[(?:FRONT|MID|END)\]|$)'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        
        sections = {"front": "", "mid": "", "end": ""}
        
        for section_name, content in matches:
            section_name = section_name.lower()
            sections[section_name] = content.strip()
        
        # If no markers found, return None
        if not any(sections.values()):
            return None
        
        return sections

    def _auto_balance_sections(self, text):
        """
        Automatically split text into sections based on sentence count.
        Aims for 25%/50%/25% distribution of SENTENCES (not tokens).
        Detail Controller will handle the actual token-based regions.
        """
        if not text or len(text.strip()) == 0:
            return {"front": "", "mid": "", "end": ""}
        
        # Split on sentence boundaries
        sentences = re.split(r'([.!?]+\s+)', text)
        
        # Recombine sentences with their delimiters
        full_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                full_sentences.append(sentences[i] + sentences[i + 1])
            else:
                full_sentences.append(sentences[i])
        if len(sentences) % 2 == 1:
            full_sentences.append(sentences[-1])
        
        # Handle single sentence or comma-separated items
        if len(full_sentences) <= 1:
            # Try splitting on commas instead
            items = [item.strip() for item in text.split(',')]
            if len(items) > 3:
                full_sentences = items
        
        total_sents = len(full_sentences)
        
        if total_sents == 0:
            return {"front": text, "mid": "", "end": ""}
        
        # Distribute: 25% / 50% / 25% of sentences
        front_count = max(1, int(total_sents * 0.25))
        mid_count = max(1, int(total_sents * 0.5))
        # Remaining go to end
        
        front_sents = full_sentences[:front_count]
        mid_sents = full_sentences[front_count:front_count + mid_count]
        end_sents = full_sentences[front_count + mid_count:]
        
        return {
            "front": " ".join(front_sents).strip(),
            "mid": " ".join(mid_sents).strip(),
            "end": " ".join(end_sents).strip()
        }

    def encode_sectioned(self, clip, mode="manual", 
                        front_text="", mid_text="", end_text="", combined_prompt="",
                        separator="comma", show_preview=True, debug=False):
        
        # Determine sections based on mode
        if mode == "manual":
            if combined_prompt:
                # Try to parse markers
                parsed = self._parse_manual_sections(combined_prompt)
                if parsed:
                    sections = parsed
                else:
                    # No markers, use separate inputs
                    sections = {
                        "front": front_text,
                        "mid": mid_text,
                        "end": end_text
                    }
            else:
                sections = {
                    "front": front_text,
                    "mid": mid_text,
                    "end": end_text
                }
        else:  # auto_balanced
            text_to_balance = combined_prompt if combined_prompt else f"{front_text} {mid_text} {end_text}"
            sections = self._auto_balance_sections(text_to_balance)
        
        # Construct final prompt
        final_prompt_parts = []
        
        if sections["front"]:
            final_prompt_parts.append(sections["front"])
        if sections["mid"]:
            final_prompt_parts.append(sections["mid"])
        if sections["end"]:
            final_prompt_parts.append(sections["end"])
        
        # Join with specified separator
        separators = {
            "comma": ", ",
            "period": ". ",
            "space": " ",
            "newline": "\n"
        }
        sep = separators.get(separator, ", ")
        final_prompt = sep.join(final_prompt_parts)
        
        # Encode - let tokenization happen naturally
        tokens = clip.tokenize(final_prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        
        conditioning = [[cond, {"pooled_output": pooled}]]
        
        # Analysis for preview
        if show_preview or debug:
            batch, seq_len, embed_dim = cond.shape
            
            # Estimate tokens for each section (preview only)
            front_est = self._estimate_tokens(sections["front"])
            mid_est = self._estimate_tokens(sections["mid"])
            end_est = self._estimate_tokens(sections["end"])
            total_est = front_est + mid_est + end_est
            
            # Calculate what Detail Controller WILL use (25%/50%/25% of actual)
            # Try to detect actual active tokens
            if total_est > 0:
                dc_front_end = int(total_est * 0.25)
                dc_mid_end = int(total_est * 0.75)
            else:
                dc_front_end = 0
                dc_mid_end = 0
            
            preview = []
            preview.append("\n" + "=" * 70)
            preview.append("FLUX.2 KLEIN SECTIONED ENCODING")
            preview.append("=" * 70)
            preview.append(f"Mode: {mode}")
            preview.append(f"Separator: {separator}")
            preview.append(f"Estimated total tokens: ~{total_est}")
            
            preview.append("\n" + "-" * 70)
            preview.append("YOUR SECTION CONTENT")
            preview.append("-" * 70)
            
            preview.append(f"\n🔵 FRONT (~{front_est} tokens)")
            preview.append(f'   "{sections["front"]}"')
            
            preview.append(f"\n🟠 MID (~{mid_est} tokens)")
            preview.append(f'   "{sections["mid"]}"')
            
            preview.append(f"\n🟢 END (~{end_est} tokens)")
            preview.append(f'   "{sections["end"]}"')
            
            preview.append("\n" + "-" * 70)
            preview.append("FINAL COMBINED PROMPT")
            preview.append("-" * 70)
            preview.append(f'"{final_prompt}"')
            
            preview.append("\n" + "=" * 70)
            preview.append("DETAIL CONTROLLER WILL USE (Dynamic)")
            preview.append("=" * 70)
            preview.append("The Detail Controller calculates regions as:")
            preview.append(f"  • FRONT region: tokens 0 to {dc_front_end} (25% of actual)")
            preview.append(f"  • MID region: tokens {dc_front_end} to {dc_mid_end} (50% of actual)")
            preview.append(f"  • END region: tokens {dc_mid_end} to {total_est} (25% of actual)")
            preview.append("\nNOTE: These are based on ACTUAL tokens after encoding,")
            preview.append("not your section boundaries. Organize semantically!")
            preview.append("=" * 70 + "\n")
            
            print("\n".join(preview))
        
        return (
            conditioning,
            sections["front"],
            sections["mid"],
            sections["end"],
            final_prompt
        )


# Node registration
NODE_CLASS_MAPPINGS = {
    "Flux2KleinSectionedEncoder": Flux2KleinSectionedEncoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2KleinSectionedEncoder": "FLUX.2 Klein Sectioned Encoder",
}