# SDS: Large Language Model-Driven Skill Discovery for Humanoid Locomotion

## Complete Project Explanation

### Project Overview

This project, **SDS (Skill Discovery System)**, represents a novel approach to automated skill acquisition for humanoid robots through the integration of large language models (LLMs) with physics-based simulation and reinforcement learning. The system leverages GPT-4-mini to automatically generate reward functions for humanoid locomotion tasks by analyzing human demonstration videos, eliminating the need for manual reward engineering - one of the most challenging aspects of reinforcement learning for robotics.

### Technical Architecture

The SDS system operates through a sophisticated multi-agent pipeline that combines computer vision, natural language processing, and robotic simulation:

**1. Video Analysis Pipeline:**
- **Input**: Human demonstration videos (e.g., walking, jumping, backflips)
- **Pose Estimation**: ViTPose (Vision Transformer-based pose estimation) extracts 2D human skeletal keypoints from demonstration videos
- **Contact Analysis**: Automated detection of foot-ground contact patterns and timing
- **Gait Pattern Recognition**: AI-driven analysis of movement phases, weight transfer, and coordination patterns

**2. Multi-Agent LLM System:**
- **Task Descriptor Agent**: Analyzes demonstration videos to identify the core locomotion task and movement objectives
- **Contact Sequence Analyser**: Examines foot contact patterns and stance/swing phase timing
- **Gait Analyser**: Identifies specific gait characteristics, coordination patterns, and movement constraints
- **Task Requirement Analyser**: Determines safety constraints, joint limits, and performance requirements
- **SUS (Skill Understanding System) Generator**: Synthesizes all analyses to generate comprehensive reward function code

**3. Physics Simulation Environment:**
- **Isaac Lab Integration**: Utilizes NVIDIA's Isaac Lab (v0.40.1) physics simulation framework
- **Humanoid Robot**: Unitree G1 humanoid robot model with full articulation (23 degrees of freedom)
- **Parallel Training**: Supports 2000-4096 parallel simulation environments for efficient policy learning
- **Real-time Physics**: GPU-accelerated physics simulation with contact dynamics and joint constraints

**4. Reward Function Generation:**
- **Code Generation**: GPT-4-mini generates complete Python reward functions based on demonstration analysis
- **Safety Integration**: Automatic inclusion of joint limits, collision avoidance, and energy constraints
- **Gait-Specific Logic**: Adaptive reward structures for different locomotion patterns (walking, jumping, acrobatic movements)
- **Progressive Complexity**: Multi-layered reward architectures with graduated objectives

**5. Iterative Improvement Loop:**
- **Policy Training**: RSL-RL (Rapid Simulation Learning - Reinforcement Learning) trains neural network policies using generated rewards
- **Performance Evaluation**: Automated video generation and analysis of learned behaviors
- **Feedback Generation**: AI-driven comparison between learned behavior and target demonstrations
- **Reward Refinement**: Iterative improvement of reward functions based on performance feedback

### Core Innovation

The fundamental innovation of SDS lies in its ability to **automatically translate human movement demonstrations into executable reward functions** without manual intervention. Traditional approaches require domain experts to manually design reward functions - a process that is time-consuming, error-prone, and requires deep understanding of both the task and the robot's dynamics. SDS eliminates this bottleneck by:

1. **Automated Reward Engineering**: Converting visual demonstrations directly into reward code
2. **Safety-First Design**: Automatically incorporating robot safety constraints and physical limitations
3. **Gait-Adaptive Logic**: Generating different reward structures for different types of locomotion
4. **Iterative Self-Improvement**: Using AI feedback to refine and improve reward functions over multiple iterations

### Technical Implementation Details

**Software Stack:**
- **Isaac Lab 0.40.1**: Physics simulation and robot modeling
- **PyTorch 2.8.0**: Neural network training and optimization
- **OpenAI GPT-4-mini**: Large language model for code generation and analysis
- **ViTPose**: Computer vision for human pose estimation
- **RSL-RL**: Reinforcement learning algorithm (Proximal Policy Optimization)

**Hardware Requirements:**
- **GPU**: NVIDIA RTX 5080 (16GB VRAM) for parallel simulation
- **RAM**: 32GB system memory for large-scale parallel training
- **Compute**: CUDA 12.8 support for GPU acceleration

**Supported Locomotion Tasks:**
- **Walking**: Natural human-like gait with proper timing and weight transfer
- **Jumping**: Vertical and dynamic jumping with controlled landing
- **Backflips**: Complex acrobatic movements requiring rotation and aerial control
- **Marching**: Ceremonial/military-style walking with specific coordination patterns

### Research Significance

This project addresses several critical challenges in robotics and AI:

1. **Reward Engineering Bottleneck**: Eliminates the need for manual reward function design
2. **Human-Robot Skill Transfer**: Enables direct learning from human demonstrations
3. **Scalable Skill Acquisition**: Provides a framework for learning diverse locomotion behaviors
4. **Safety-Aware Learning**: Automatically incorporates robot safety constraints
5. **Cross-Modal Understanding**: Bridges the gap between visual demonstrations and executable robot behaviors

### Evaluation Methodology

The system's performance is evaluated through multiple metrics:
- **Task Completion**: Successful execution of demonstrated locomotion patterns
- **Movement Quality**: Similarity to human demonstrations in terms of gait timing, coordination, and smoothness
- **Safety Compliance**: Adherence to joint limits, collision avoidance, and energy constraints
- **Learning Efficiency**: Speed of convergence and sample efficiency during training
- **Generalization**: Ability to adapt learned skills to variations in terrain and conditions

---

## Master's Thesis Introduction

### Project Title
**"Automated Skill Discovery for Humanoid Locomotion: A Large Language Model-Driven Approach to Reward Function Generation"**

**Student:** [Your Name]  
**Supervisor:** [Supervisor Name]  
**Institution:** [University Name]  
**Program:** Master of Science in [Robotics/AI/Computer Science]

### Motivation

The development of versatile humanoid robots capable of natural locomotion remains one of the most challenging frontiers in robotics. While significant advances have been made in hardware design and control algorithms, the process of teaching robots to move like humans continues to require extensive manual engineering and domain expertise. Traditional approaches to robotic skill acquisition through reinforcement learning face a critical bottleneck: the design of appropriate reward functions that can guide learning toward desired behaviors while maintaining safety and efficiency.

The importance of solving this challenge extends far beyond academic research. Humanoid robots with natural locomotion capabilities have transformative potential across numerous sectors of society. In healthcare, such robots could provide mobility assistance to elderly individuals, support rehabilitation therapy, and operate in medical environments where human-like dexterity and movement are essential. In disaster response scenarios, humanoid robots capable of navigating human-designed environments could perform rescue operations in situations too dangerous for human responders. In manufacturing and service industries, robots with human-like locomotion could work alongside humans in shared spaces, adapting to environments designed for human movement without requiring specialized infrastructure modifications.

Furthermore, the broader implications for artificial intelligence research are significant. Successfully bridging the gap between human demonstrations and robotic skill acquisition represents a crucial step toward more intuitive human-robot interaction and collaborative robotics. The ability to teach robots through demonstration rather than explicit programming could democratize robotics, making advanced robotic capabilities accessible to users without specialized technical knowledge.

### The Challenge

The fundamental challenge in teaching humanoid robots to walk, jump, or perform complex locomotion lies in translating human intuitive understanding of movement into mathematical formulations that can guide machine learning algorithms. Traditional reinforcement learning approaches require carefully crafted reward functions that specify exactly what constitutes good or bad behavior at each moment during a locomotion task. This reward engineering process is notoriously difficult for several reasons.

First, human locomotion involves complex multi-objective optimization that balances competing requirements such as energy efficiency, stability, speed, and safety. Capturing this complexity in a mathematical reward function requires deep understanding of biomechanics, robot dynamics, and optimization theory. Second, locomotion behaviors exhibit hierarchical structure - successful walking requires coordination between joint-level control, limb-level coordination, and whole-body balance - making it challenging to design rewards that operate effectively across multiple temporal and spatial scales.

Previous research efforts have attempted to address these challenges through various approaches. Hand-crafted reward functions based on biomechanical principles have achieved some success but require extensive domain expertise and often fail to generalize across different robots or tasks. Imitation learning approaches that directly copy human demonstrations can reproduce specific movements but struggle with adaptation to new situations or recovery from perturbations. Evolutionary approaches that automatically discover reward functions through trial and error require prohibitively long computation times and often converge to unnatural or unsafe movement patterns.

Recent advances in large language models have demonstrated remarkable capabilities in code generation, reasoning about complex systems, and understanding multimodal information. However, the application of these capabilities to robotic skill acquisition remains largely unexplored, particularly in the domain of locomotion where safety constraints and physical realism are paramount.

### Project Aims and Scope

This research aims to develop and evaluate a novel system for automated skill discovery in humanoid locomotion that leverages large language models to generate reward functions directly from human demonstration videos. The specific objectives of this project are:

**Primary Objective:** To design and implement an automated pipeline that can analyze human locomotion demonstrations and generate executable reward functions for training humanoid robots in physics simulation, without requiring manual reward engineering or domain-specific programming.

**Secondary Objectives:**

1. **Develop a multi-agent LLM system** capable of understanding locomotion tasks from visual demonstrations, incorporating computer vision for pose estimation, gait analysis, and movement pattern recognition.

2. **Create comprehensive prompt engineering frameworks** that guide LLMs to generate safe, efficient, and task-appropriate reward functions while automatically incorporating robot safety constraints and physical limitations.

3. **Establish evaluation methodologies** for assessing the quality of generated reward functions in terms of task completion, movement naturalness, safety compliance, and learning efficiency.

4. **Demonstrate generalizability** across multiple locomotion tasks including walking, jumping, and acrobatic movements using a consistent framework and methodology.

5. **Validate the approach** through extensive simulation experiments using the Unitree G1 humanoid robot in Isaac Lab physics simulation, measuring performance against both human demonstrations and manually engineered baselines.

The scope of this research is focused on humanoid locomotion in flat terrain environments, using the Unitree G1 robot as the primary experimental platform. While the methodology is designed to be generalizable, this project specifically addresses bipedal locomotion tasks that can be demonstrated by humans and learned through reinforcement learning in simulation.

### Approach and Methodology

This research employs a novel methodology that combines computer vision, natural language processing, and reinforcement learning in an integrated framework for automated skill discovery. The approach differs from previous work by treating reward function generation as a code synthesis problem that can be solved through large language model capabilities, rather than as an optimization or search problem.

**Multi-Agent Video Analysis:** The system begins by analyzing human demonstration videos using a cascade of specialized AI agents. ViTPose, a state-of-the-art vision transformer-based pose estimation system, extracts detailed human skeletal information from demonstration videos. This visual data is then processed by multiple GPT-4-mini agents, each specialized for different aspects of locomotion analysis: task identification, contact pattern analysis, gait characterization, and safety requirement determination.

**Prompt Engineering for Code Generation:** A critical innovation of this approach is the development of comprehensive prompt engineering frameworks that guide the language model to generate not just reward functions, but complete Python code modules that integrate seamlessly with physics simulation environments. These prompts incorporate detailed knowledge about robot dynamics, safety constraints, and locomotion biomechanics, enabling the LLM to generate code that is both functionally correct and physically realistic.

**Iterative Refinement Through AI Feedback:** Unlike traditional approaches that rely on human experts to evaluate and refine reward functions, this system employs automated feedback loops where AI agents analyze the resulting robot behaviors and suggest improvements to the reward functions. This creates a self-improving system that can refine its approach based on observed outcomes.

**Safety-First Design Philosophy:** A key differentiator of this approach is the automatic incorporation of safety constraints into generated reward functions. Rather than treating safety as an afterthought, the system is designed to ensure that all generated rewards include appropriate joint limits, collision avoidance, and energy constraints from the outset.

This methodology represents a fundamental shift from previous approaches by treating the entire skill acquisition pipeline - from demonstration analysis to reward function deployment - as an integrated, automated system rather than a collection of separate manual steps. The approach leverages the reasoning capabilities of modern large language models while grounding their outputs in the physical realities of robotic systems and the mathematical requirements of reinforcement learning algorithms.

Through this integrated approach, the research aims to demonstrate that automated skill discovery for humanoid locomotion is not only feasible but can achieve performance levels comparable to or exceeding manually engineered approaches while requiring significantly less human expertise and intervention. 