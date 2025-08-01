import time
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Sample collection of texts (could be paragraphs, sections, or full docs)
docs = [
"of common standards for the preservation and archiving local digital content",
        "of a common interface to ODL resources and services that can be imbedded in local",
        "library web sites",
        "What could the ODL really mean?",
        "For each Ontario citizen it could mean:",
        "One local point of entry to access seamless electronic library services and",
        "resources for their personal, educational and professional needs;",
        "Access to credible, high-quality, user-friendly electronic services through their",
        "community, school, or academic library;",
        "Confidence that the electronic services and sources they \u2013 and their children",
        "\u2013 are using are safe, valid, and bringing them both global and local",
        "perspectives;",
        "Electronic information and tools that enhance job skills and the learning",
        "experience",
        "For each Ontario student it could mean:",
        "One local point of entry to quality, curriculum- based electronic information",
        "resources and services",
        "Connection to their individual educational environment, whether at the",
        "elementary, secondary or post-secondary levels",
        "RFP: To Develop the Ontario Digital Library Business Plan",
        "March 2003",
        "5",
        "Confidence that the services and resources people are using are credible,",
        "available when they need them and adaptable to different learning styles",
        "For each Ontario library it could mean:",
        "One point of entry for themselves and their clientele to quality electronic",
        "resources and services that support and enhance their collections, programs",
        "and services",
        "The opportunity to show case their local community, collections, services, and",
        "expertise",
        "The opportunity to gain from the provincial, collaborative partnership in order",
        "to provide a seamless gateway to digital resources and services",
        "For the Ontario government it could mean:",
        "A  point of entry to information services and resources for all Ontario citizens,",
        "connecting to portal initiatives .",
        "Support of the Ontario E-Learning strategy through a collaborative digital",
        "library initiative that positions Ontario on the global e-learning scene",
        "Improved access and flexibility for all students at all ages and stages",
        "(supports Ontario\u2019s lifelong learning strategy)",
        "The opportunity to address emerging job skills shortages and facilitate school",
        "to work and job to job transitions with the necessary information and",
        "knowledge supports and tools",
        "ODL will be an incorporated non-profit organization governed by a Board of Directors elected",
        "by a voting membership. The Board will include representatives from all stakeholders. We",
        "envision a governance model similar to that prepared for the province\u2019s ORION network.",
        "More information regarding the envisioned phasing, funding and resources required for the",
        "ODL can be found in the appendixes.",
        "The Business Plan to be Developed",
        "The business plan which needs to be developed for the ODL must be a formal business plan",
        "that documents and clearly communicates the ODL\u2019s services, funding and governance",
        "structures, as well as implementation plans for 2004-2005, and operational plans for 2005-",
        "2007. The planning process must also secure the full commitment of all stakeholders, as",
        "represented on the Steering Committee.",
        "Specifically, the business plan must include:",
        "how the ODL will be implemented, including the timeline",
        "the financial plan for the implementation",
        "the financial plan for the first 2 operating years, including capital and operating costs,",
        "revenues, etc.",
        "a financial forecast for the succeeding 2 operating years",
        "the services and products to be delivered by the ODL",
        "how the ODL will operate and be managed following the implementation",
        "who will be involved, and what their role/responsibility will be, for both the",
        "implementation and operational stages",
        "RFP: To Develop the Ontario Digital Library Business Plan",
        "March 2003",
        "6",
        "the marketing and communications plan for the ODL",
        "the commitment of all stakeholders to their responsibilities",
        "The process of developing this business plan must be extremely consultative to ensure that",
        "all stakeholders are engaged in creating a synergistic ODL organization.  The proposal must",
        "indicate how this consultative process will be approached.",
        "The business plan for the ODL must address significant issues. There are, for example,",
        "enormous differences in the financial resources available to libraries mandated to provide",
        "similar services. Some post-secondary and public libraries, particularly those in large urban",
        "areas, have the facilities, funding and technological infrastructure necessary to service their",
        "patrons with electronic services and resources.  Many others, particularly in rural regions, do",
        "not.  The proposal must indicate how these issues will be approached.",
        "Milestones",
        "1) A preliminary report will be issued during June 2003.",
        "2) It is expected that an Interim Report, suitable for distribution to the broader library",
        "community will be available by August 1, 2003 and that there will be an opportunity",
        "for responses to be evaluated.",
        "3) The business plan must be completed and approved by the ODL Steering Committee",
        "no later than September 30, 2003.",
        "Approach and Specific Proposal Requirements",
        "The firm/consultant  (or proposed team of consultants)  will be expected to work closely with",
        "the ODL Steering Committee. Terms of reference for the Committee are in the appendix.",
        "Given the consultative nature of this business planning process, the firm/consultant will also",
        "be expected to travel and communicate regularly with various stakeholders as well as with",
        "electronic resource publishers/vendors.",
        "The proposal should include the following information:",
        "a) name of the firm/consultant",
        "b) names of those individuals who will be engaged in this project, their specific",
        "responsibilities on this project and relevant experience/qualifications",
        "c) description of similar engagements that highlight the firm\u2019s experience in business",
        "planning and building stakeholder commitment",
        "d) references with details of work completed for these references",
        "e) description of the approach that will be used for completing the business plan,",
        "including a timeline",
        "f) cost to complete the study including estimated expenses (i.e.: travel, etc.) and payment",
        "structure",
        "RFP: To Develop the Ontario Digital Library Business Plan",
        "March 2003",
        "7",
        "Evaluation and Awarding of Contract",
        "The contract will be awarded to the bidder whose submission offers the best value; the contract will",
        "not necessarily be awarded to the lowest bidder. We reserve the right not to award the contract to",
        "any of the bidders responding to this RFP and we may seek further response.",
        "Specifically, proposals will be evaluated proposals according to the following criteria:",
        "Quality of the proposal /approach outlined for undertaking the business planning process",
        "Demonstrated experience",
        "Cost, including expenses",
        "Timeline and projected completion date",
        "Other relevant factors as determined by the ODL Steering Committee",
        "Questions regarding this RFP should be directed by e-mail only to Michael Ridley",
        "(mridley@uoguelph.ca).",
        "Bidders are asked not to contact any other member of the ODL Catalyst Team or the ODL Steering",
        "Committee.",
        "RFP: To Develop the Ontario Digital Library Business Plan",
        "March 2003",
        "8",
        "Appendix A:  ODL Envisioned Phases & Funding"
]

start = time.time()

# Configure TF-IDF
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),         # unigrams and bigrams
    max_features=20             # top 20 features (can increase this)
)
X = vectorizer.fit_transform(docs)

# Extract top keywords per doc
feature_names = vectorizer.get_feature_names_out()
top_k = 5

for doc_idx, doc_vector in enumerate(X):
    print(f"\n🔹 Document {doc_idx + 1}:")
    sorted_indices = np.argsort(doc_vector.toarray()[0])[::-1]
    for idx in sorted_indices[:top_k]:
        term = feature_names[idx]
        score = doc_vector.toarray()[0][idx]
        print(f"{term} - {score:.4f}")

end = time.time()
print(f"\n✅ Total processing time: {end - start:.4f} seconds")
