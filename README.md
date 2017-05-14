## Problem Statement 

Financial fraud is a problem that affects the entire world economy.  Despite the importance of fully understanding it, according the [doctoral thesis](http://bth.diva-portal.org/smash/record.jsf?pid=diva2%3A955852&dswid=4451) of Edgar Alonso Lopez-Rojas, there is a general lack of understanding of fraudulent behavior in financial domains today.

We lack understanding primarily due to our inability to research the subject and we cannot research the subject because of our lack of financial transaction data.  To help to address this situation two simulation models (Payment Simulator (PaySim) and Retail Store Simulator (RetSim)) were developed to generate fictional transactional data, of a fraudulent and non-fraudulent nature.  Both behaviors, fraudulent and non-fraudulent, wsa codified in the simulators using available data with the intent to apply machine learning methodologies in hopes to better understand and stop financial fraud.

We will focus on a modified version of the [PaySim dataset on Kaggle](https://www.kaggle.com/ntnu-testimon/paysim1) that Edgar was so kind to have shared with the rest of us.  It is through using this dataset, applying methods of scaling, featuring engineering, and machine learning models that we will begin to better understand the behaviors that serve as leading indicators that a transaction is fraudulent.  Only then will we be better positioned to protect the global financial interests of it's law abiding citizens.

![eye](https://cloud.githubusercontent.com/assets/22734960/26037132/1547440a-38bb-11e7-9616-59c5651b1d34.png)

## An Exploratory Data Analysis

A methodical EDA (exploratory data analysis) is comprised of four steps.

1. The first of these steps involves loading the data into memory and storing it in a dataframe. When a dataset does not fit into memory there are alternative methods of doing this - though I am unsure of what those are.
2. In step two we will make the data set a "tidy data" set by following the guidelines established by [Hadley Wickham](http://vita.had.co.nz/papers/tidy-data.html)
3. Step three will use descriptive statistics and exploratory visualizations to understand the data at a macro level
4. In step four we will aggregate the data and explore the group properties in a much more detailed manner



## Data Preprocessing

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/hwkkix/DetectingFraud/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
