![Open-H Initiative](assets/open_h_header.jpg)

<div align="center">

# Open-H Initiative: Data Contribution How-To Guide

[![Discord](https://img.shields.io/badge/Discord-Join%20our%20community-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/YZEhNcTHtc)
[![LeRobot](https://img.shields.io/badge/LeRobot-v0.3.3-FF6B6B?style=for-the-badge)](https://docs.phospho.ai/learn/lerobot-dataset)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-4CAF50?style=for-the-badge)](https://creativecommons.org/licenses/by/4.0/)
[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-live-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment)

</div>

> [!IMPORTANT]
> **Open-H-Embodiment v1 is now live on Hugging Face:**  
> https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment
>
> This first public release provides the initial Open-H-Embodiment dataset in **LeRobot v2.1** format for healthcare robotics research and development.

This guide provides a comprehensive overview of how to contribute meaningful data to the Open-H initiative, ensuring consistency and quality across all contributions.

## 📦 Dataset Release

**Open-H-Embodiment v1 is now available on Hugging Face:**  
https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment

This release marks the first public version of the Open-H-Embodiment dataset, a community-driven collection of paired video and kinematics data for healthcare robotics.

The Hugging Face dataset page is the primary location for:
- accessing released dataset assets,
- reviewing dataset metadata and documentation,
- tracking public dataset versions.

This GitHub repository remains the primary location for:
- contribution instructions,
- formatting requirements,
- conversion scripts,
- synchronization tools,
- validation utilities.

## How to Participate

1. **Review the Request for Proposals**  
   Read the [Open-H RFP](assets/open-h-rfp.pdf) to confirm your proposed dataset aligns with the initiative. The RFP outlines the technical scope, eligibility, and evaluation criteria for the one-page submission reviewed by the Open-H-Embodiment steering committee.

2. **Prepare and Submit Your Proposal**  
   Develop a concise one-page summary describing the dataset, collection methodology, and specific tasks. Follow the instructions in the RFP and submit the document for steering committee review.

3. **Upload Your Approved Dataset**  
   After approval, upload your data to the [Open-H shared drive](https://drive.google.com/drive/folders/1fenrjbsSYaeLz-U_LD7K063oT2el8ueX?usp=sharing).

   A dedicated folder will be provisioned for your institution (and each participating lab, if applicable) to keep contributions organized.

4. **Register Dataset Details**  
   Record the dataset metadata, documentation links, and key contacts in the [dataset tracking sheet](https://docs.google.com/spreadsheets/d/1vG9778S6G-Embum9ZjK_NlZGR0KFa2VkMnGDnZB-Exk/edit?usp=sharing).

   This ensures the community can discover and integrate your contribution.

5. **Inclusion in Future Releases**  
   Approved and properly formatted contributions may be incorporated into future public releases of Open-H-Embodiment on Hugging Face, subject to review for quality, documentation completeness, and licensing compliance.

## 🚀 LeRobot Installation

Before using the conversion scripts and following this dataset preparation guide, install the required version of LeRobot:

### Required Version: LeRobot v0.3.3

```bash
pip install lerobot==0.3.3
