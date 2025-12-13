Credit Scoring Business Understanding
Basel II, Risk Measurement, and Model Interpretability

The Basel II Capital Accord places strong emphasis on accurate risk measurement, transparency, and governance in credit decision-making. Under Basel II, banks are required not only to quantify credit risk (e.g., Probability of Default) but also to demonstrate how those risk estimates are produced and validated.

As a result, our credit scoring model must be interpretable, well-documented, and auditable. This ensures that internal risk teams, regulators, and external auditors can understand the assumptions, features, and logic behind credit decisions. In the context of a Buy-Now-Pay-Later (BNPL) service, where decisions are automated and made in real time, explainability is critical to justify approvals, rejections, and credit limits while remaining compliant with regulatory expectations.

Proxy Default Variable: Motivation and Business Risks

In this project, we do not have an explicit loan default label because the dataset originates from an eCommerce platform rather than a traditional lending system. Therefore, creating a proxy default variable is necessary to approximate credit risk using observable customer behavior.

Behavioral indicators such as Recency, Frequency, and Monetary (RFM) patterns are used to infer whether a customer is likely to default. This enables the development of a predictive credit risk model in the absence of formal repayment history.

However, relying on a proxy introduces business risks:

The proxy may not perfectly represent true default behavior, leading to misclassification.

Customers may be incorrectly labeled as high-risk, resulting in lost revenue and customer dissatisfaction.

Conversely, underestimating risk may increase default rates and financial losses.

To mitigate these risks, proxy definitions must be carefully designed, validated, and continuously monitored to ensure alignment with real-world repayment outcomes.

Trade-offs Between Interpretable and Complex Models

In a regulated financial environment, there is a fundamental trade-off between model interpretability and predictive performance.

Simple, interpretable models (e.g., Logistic Regression with Weight of Evidence encoding):

Are transparent and easy to explain to regulators and business stakeholders.

Support clear reasoning about how features influence credit decisions.

Are more suitable for regulatory compliance and audit requirements.

May have lower predictive power compared to complex models.

Complex, high-performance models (e.g., Gradient Boosting):

Often achieve higher accuracy and better risk separation.

Can capture non-linear relationships in customer behavior.

Are harder to interpret and explain, increasing regulatory and governance challenges.

Require additional tools and documentation to justify their use in production.

In this project, model selection must balance predictive accuracy with explainability, ensuring the final solution is both business-effective and regulatory-compliant.
