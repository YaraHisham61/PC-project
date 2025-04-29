// #include <vector>
// #include <string>
// #include <unordered_map>
// #include <memory>

// class PhysicalOperatorNode
// {
// protected:
//     std::string name;
//     std::unordered_map<std::string, std::string> parameters;
//     std::vector<std::unique_ptr<PhysicalOperatorNode>> children;
//     int indent_level;

// public:
//     PhysicalOperatorNode(const std::string &op_name,
//                          const std::unordered_map<std::string, std::string> &params,
//                          int indent)
//         : name(op_name), parameters(params), indent_level(indent) {}

//     virtual ~PhysicalOperatorNode() = default;

//     void addChild(std::unique_ptr<PhysicalOperatorNode> child)
//     {
//         children.push_back(std::move(child));
//     }

//     const std::string &getName() const { return name; }
//     const std::unordered_map<std::string, std::string> &getParameters() const { return parameters; }
//     const std::vector<std::unique_ptr<PhysicalOperatorNode>> &getChildren() const { return children; }
//     int getIndentLevel() const { return indent_level; }

//     virtual void print() const
//     {
//         std::string pad(indent_level, ' ');
//         std::cout << pad << "- " << name << "\n";
//         if (!parameters.empty())
//         {
//             std::cout << pad << "  Params: ";
//             for (const auto &pair : parameters)
//             {
//                 std::cout << pair.first << ": " << pair.second << ", ";
//             }
//             std::cout << "\n";
//         }
//         for (const auto &child : children)
//         {
//             child->print();
//         }
//     }
// };

// // Derived classes for specific operator types
// class SeqScanNode : public PhysicalOperatorNode
// {
// public:
//     SeqScanNode(const std::unordered_map<std::string, std::string> &params, int indent)
//         : PhysicalOperatorNode("SEQ_SCAN", params, indent) {}

//     void print() const override
//     {
//         PhysicalOperatorNode::print();
//         // Add SEQ_SCAN specific printing if needed
//     }
// };

// class FilterNode : public PhysicalOperatorNode
// {
// public:
//     FilterNode(const std::unordered_map<std::string, std::string> &params, int indent)
//         : PhysicalOperatorNode("FILTER", params, indent) {}
// };

// class ProjectionNode : public PhysicalOperatorNode
// {
// public:
//     ProjectionNode(const std::unordered_map<std::string, std::string> &params, int indent)
//         : PhysicalOperatorNode("PROJECTION", params, indent) {}
// };

// class OrderByNode : public PhysicalOperatorNode
// {
// public:
//     OrderByNode(const std::unordered_map<std::string, std::string> &params, int indent)
//         : PhysicalOperatorNode("ORDER_BY", params, indent) {}
// };

// // Factory function to create the appropriate node type
// std::unique_ptr<PhysicalOperatorNode> createOperatorNode(
//     const std::string &name,
//     const std::unordered_map<std::string, std::string> &params,
//     int indent)
// {

//     if (name == "SEQ_SCAN")
//     {
//         return std::make_unique<SeqScanNode>(params, indent);
//     }
//     else if (name == "FILTER")
//     {
//         return std::make_unique<FilterNode>(params, indent);
//     }
//     else if (name == "PROJECTION")
//     {
//         return std::make_unique<ProjectionNode>(params, indent);
//     }
//     else if (name == "ORDER_BY")
//     {
//         return std::make_unique<OrderByNode>(params, indent);
//     }
//     else
//     {
//         return std::make_unique<PhysicalOperatorNode>(name, params, indent);
//     }
// }

// // Modified printPhysicalPlan to build the tree structure
// std::unique_ptr<PhysicalOperatorNode> buildPhysicalPlanTree(duckdb::PhysicalOperator *op, int indent)
// {
//     if (!op)
//     {
//         return nullptr;
//     }

//     // First build children
//     std::vector<std::unique_ptr<PhysicalOperatorNode>> children;
//     for (auto &child : op->children)
//     {
//         children.push_back(buildPhysicalPlanTree(&(child.get()), indent + 2));
//     }

//     // Get operator info
//     std::string name = op->GetName();
//     auto params = op->ParamsToString();

//     // Create the node
//     auto node = createOperatorNode(name, params, indent);

//     // Add children
//     for (auto &child : children)
//     {
//         if (child)
//         {
//             node->addChild(std::move(child));
//         }
//     }

//     return node;
// }

// // Usage example:
// void processPhysicalPlan(duckdb::PhysicalOperator *root)
// {
//     auto plan_tree = buildPhysicalPlanTree(root, 0);
//     if (plan_tree)
//     {
//         plan_tree->print();

//         // Now you have the entire plan in a tree structure that you can analyze
//         // For example, you can access specific nodes, their parameters, etc.
//     }
// }