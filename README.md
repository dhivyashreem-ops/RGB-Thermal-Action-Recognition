@'

\# CMAF-Net: Bidirectional Cross-Modal Attention Fusion for RGB-Thermal Action Recognition



\## Overview



This repository contains the implementation of \*\*CMAF-Net\*\*, a bidirectional Cross-Modal Attention Fusion framework for RGB-thermal human action recognition.



The proposed framework learns interactions between RGB and thermal feature representations using bidirectional cross-modal attention:



\- RGB -> Thermal

\- Thermal -> RGB



The system operates on pre-extracted 512-dimensional RGB and thermal feature representations and performs 27-class action recognition.



\## Key Features



\- RGB-thermal multimodal action recognition

\- 512-dimensional RGB and thermal input features

\- 256-dimensional shared latent representation

\- 16 feature tokens per modality

\- 4-head bidirectional cross-modal attention

\- Residual attention enhancement

\- Learnable positional embeddings

\- Class-balanced weighted cross-entropy

\- Stratified train/validation split

\- Early stopping based on validation Macro-F1

\- Three-seed reproducibility evaluation

\- Cross-modal attention entropy analysis

\- Per-class performance analysis



\## Dataset



The experiments use the \*\*DarkAct RGB-thermal action-recognition dataset\*\*.



DarkAct contains paired RGB-thermal video data covering 27 human action categories.



This repository uses a processed feature representation rather than directly training CMAF-Net on raw video frames.



\### Processed Features



| Split | RGB Features | Thermal Features |

|---|---:|---:|

| Training | 8,769 x 512 | 8,769 x 512 |

| Testing | 4,009 x 512 | 4,009 x 512 |



The 8,769 training samples and 4,009 test samples refer to the processed feature pipeline used in this implementation.



\## CMAF-Net Architecture



The network performs:



```text

RGB 512-D -> Projection 256-D -> 16 RGB Tokens

&#x20;                                     |

&#x20;                                     v

&#x20;                             RGB -> Thermal

&#x20;                             Cross-Attention

&#x20;                                     |

&#x20;                                     v

&#x20;                             Residual + Norm

&#x20;                                     |

&#x20;                                     v

&#x20;                               Token Pooling

&#x20;                                     |

&#x20;                                     v

&#x20;                             RGB Vector 256-D





Thermal 512-D -> Projection 256-D -> 16 Thermal Tokens

&#x20;                                     |

&#x20;                                     v

&#x20;                             Thermal -> RGB

&#x20;                             Cross-Attention

&#x20;                                     |

&#x20;                                     v

&#x20;                             Residual + Norm

&#x20;                                     |

&#x20;                                     v

&#x20;                               Token Pooling

&#x20;                                     |

&#x20;                                     v

&#x20;                          Thermal Vector 256-D



&#x20;             RGB Vector + Thermal Vector

&#x20;                        |

&#x20;                        v

&#x20;                   Fusion MLP

&#x20;                   512 -> 256

&#x20;                        |

&#x20;                        v

&#x20;                  Classifier

&#x20;                256 -> 128 -> 27

&#x20;                        |

&#x20;                        v

&#x20;                 Action Prediction

