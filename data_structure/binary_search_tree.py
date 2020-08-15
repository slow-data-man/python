# -*- coding: utf-8 -*-
# @Time    : 2020/8/9 16:27
# @Author  : wangxg
from data_structure.base.node import TreeNode
from data_structure.binary_tree import BinaryTree


class BinarySearchTree(BinaryTree):
    def __init__(self, root=None):
        self.root = root

    def add(self, value):
        """二叉搜索树添加元素"""
        node = TreeNode(value)
        if self.root is None:
            self.root = node
            return
        queue = [self.root]
        while queue:
            cur_node = queue.pop(0)
            if node.value <= cur_node.value:
                if cur_node.left is None:
                    cur_node.left = node
                    return
                else:
                    queue.append(cur_node.left)
            else:
                if cur_node.right is None:
                    cur_node.right = node
                    return
                else:
                    queue.append(cur_node.right)


if __name__ == '__main__':
    tree = BinarySearchTree()
    for i in [8, 7, 6, 6, 12, 65, 3, 2, 1]:
        tree.add(i)
    lst = tree.breadth_travel()
    print(lst)
