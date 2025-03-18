# Specifications

In this section we write the expected functions that the MIPHA platform should offer.
Functions will be specified as "user stories" ("As a user, I want to..."), noted US.

## Data import

When it comes to disease prediction, data is usually formatted as 2D or 3D matrices. The goal of this part of the app is
not to store data like a database would, but to classify datasets that have been created for MIPHA implementations.

### [US 1.1] As a user, I want to store my data on the MIPHA platform

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

### [US 1.2] As a user, I want to use my data with MIPHA implementations

- Uploaded data is converted into Data Sources (MIPHA object) using the information provided during the upload.

### [US 1.3] As a user, I want to get insights on my data

- On any given data source, the user can get insights on the data: dimension of data, amount of missing data,
  distribution of features, min / max / mean / var / std, etc.
- When the insights have been computed, they are saved along with the metadata (no need to compute them again).

## Running and evaluating implementations

### [US 2.1] As a user, I want to evaluate my MIPHA implementation on stored data

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

## Archiving implementations and experiments

### [US 3.1] As a user, I want to store my MIPHA implementations on the platform

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

### [US 3.2] As a user, I want to store the results of my experiments

- When an experiment is run, results output by the Evaluator should be saved, along with comments written by the user.
- An experiment should contain as much information as possible to be reproducible: components and models used,
  hyperparameters, data sources / datasets, etc.
- An interface lets users check the history of experiments and all the information saved along with them.
