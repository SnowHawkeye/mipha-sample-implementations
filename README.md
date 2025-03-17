# Sample Implementations of the MIPHA Framework

## Introduction

This document specifies the expected deliverables for the G1-G2 MIPHA project @CentraleLille.

The repository contains data and code samples that can be used to test the MIPHA platform with regard to the expected
functions described in the sections below.

## The MIPHA framework

### General presentation

MIPHA stands for Modular data Integration for Predictive Healthcare Analysis.
It is a Python framework that allows users to create machine learning models, taking into account several data sources.
In other words, people who use the [MIPHA library](https://github.com/SnowHawkeye/mipha) can create their own
implementation
of the MIPHA components to create their own prediction model.

**Why use this prediction framework?**

Its features are as follows:

- [A] Flexible framework allowing for the study of any disease
- [B] Ability to include data from various sources
- [C] Modular architecture designed for reusability

Using the MIPHA framework allows for the easy inclusion or exclusion of data sources when performing predictions. Its
modular architecture allows enables easy reusability of the components.

Finally — and this is the purpose of the associated G1-G2 project — the formalization of this framework allows its
integration into applications and APIs.

### MIPHA components

The following diagram show the components of a MIPHA implementation.

![mipha_architecture.png](media/mipha_architecture.png)

- **Data Sources**: They contain the data processed by the model. They are not technically part of the "model" itself,
  but MIPHA implementations require the data to be formalized as Data Sources in order to function properly. On top of
  the data itself, Data Sources contain a "data type" property, which is used by the Feature Extractors.
- **Feature extractors**: Their role is to use Data Sources to generate features usable by the machine learning model (
  kernel). For example, they can perform feature selection, dimension reduction, vectorization... or nothing at all (and
  give the raw data to the rest of the framework). Each feature extractor specifies which data types it can handle: for
  example, a neural network-based feature extractor may be able to handle the "laboratory" and "electrocardiogram" data
  types. This means that some feature extractors may run on several data sources.
- **Aggregator**: The aggregator collects the data produced by feature extractors and regroups it into a single matrix,
  usable by the kernel. It can be a simple concatenation, or a more complex process involving further processing.
- **Kernel**: The kernel of a MIPHA model is the machine learning model at its core. It performs the prediction using
  the data fed by the aggregator.
- **Evaluator**: It uses the results provided by the Kernel to evaluate its performance.

All of these components are very flexible: there are few constraints on what one can or cannot implement. Again, the
main point of using MIPHA is to clarify and modularize the structure of a model.

## The MIPHA platform

The MIPHA platform is the expected deliverable of the MIPHA G1-G2 project.
The goal is to create an application that allows users of the MIPHA framework to:

- Import and store their data, and get general insights on it.
- Run and evaluate their implementations.
- Store and document both their implementations and the experiments they realized with them.

This platform is complementary to the MIPHA framework: it uses the framework's modularity to its advantage, and allows
users to keep track of their implementations and the performance they obtained with each component.
This should help researchers create reproducible results, and facilitates communication and knowledge transfer.

## Specifications

In this section we write the expected functions that the MIPHA platform should offer.
Functions will be specified as "stories" ("As a user, I want to...")

### Data import

When it comes to disease prediction, data is usually formatted as 2D or 3D matrices. The goal of this part of the app is
not to store data like a database would, but to classify datasets that have been created for MIPHA implementations.

#### [US 1.1] As a user, I want to store my data on the MIPHA platform

⚠️ It is important to understand that data sources are not necessarily independent. For example, a set of data sources
may need to have the same dimensions. As a consequence, users should be allowed to upload several "data sources" at
once, and they should be stored in the same group (that we can call a "data set"). **This vocabulary distinction is very
important, and it is essential to take it into account in the app.**

- Data will be given as numpy arrays saved in `.pkl` (pickle) format. For now, we focus on 3D data.
- When uploading the data, the user can optionally specify what the columns/features correspond to (manually or with an
  additional file).
- When uploading the data, the user has to provide information necessary to generate the corresponding data sources and
  data sets (data type, description of the data, etc.).
- The uploaded data is stored with the associated metadata and can be browsed.

#### [US 1.2] As a user, I want to use my data with MIPHA implementations

- Uploaded data is converted into Data Sources (MIPHA object) using the information provided during the upload.

#### [US 1.3] As a user, I want to get insights on my data

- On any given data source, the user can get insights on the data: dimension of data, amount of missing data,
  distribution of features, min / max / mean / var / std, etc.
- When the insights have been computed, they are saved along with the metadata (no need to compute them again).

### Running and evaluating implementations

#### [US 2.1] As a user, I want to evaluate my MIPHA implementation on stored data

- MIPHA implementations can be uploaded on the platform (see next section). The user can select one of the uploaded
  implementations to run an experiment.
- The user can also pick data from one or several of the stored data sources.
- The platform runs the chosen implementation on the chosen data.
- Experimental results (computed by the Evaluator) are displayed and store (see next section).

⚠️ It is important to remember that the Evaluator is developed by the user. The results it outputs can technically be of
any kind. A solution to this issue would be to create a custom implementation of the Evaluator, specific to the MIPHA
platform (for example, ensuring that results are output in a JSON file, or that curves are stored in a certain format).
Users who want to display their results should implement this MiphaPlatformEvaluator.

⚠️ The risks associated with running user-written code directly on the platform should be taken into account. Not only
security risks, but also potential crashes, etc.

### Archiving implementations and experiments

#### [US 3.1] As a user, I want to store my MIPHA implementations on the platform

- The user can upload implementations of individual components, which should be stored along with metadata: name,
  notes / comments, type of component, nature of the hyperparameters.
- The user can create a MIPHA model, which is an assembly of components.
- Similarly to before, it is worth considering the creation of a custom superclass for MIPHA platform components, in
  order to force a certain formatting of the data.
- Upon uploading, checks are run to verify that the components are indeed MIPHA implementations usable by the platform,
  and that they do not present any risk.
- An interface lets users check existing components and models, and the experiments and datasets they are linked to.

⚠️ We are uploading code to the platform that is meant to be executed later on. It is **vital** to research if / how it
is possible to do this in a secure way. Without validation or control of any kind, a hacker could easily take control of
the platform.

#### [US 3.2] As a user, I want to store the results of my experiments

- When an experiment is run, results output by the Evaluator should be saved, along with comments written by the user.
- An experiment should contain as much information as possible to be reproducible: components and models used,
  hyperparameters, data sources / datasets, etc.
- An interface lets users check the history of experiments and all the information saved along with them.